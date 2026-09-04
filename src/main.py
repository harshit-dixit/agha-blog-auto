from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import DB_PATH, TOPICS_FILE, get_settings
from src.catalog import load_catalog
from src.db.history import HistoryDB
from src.generators.gaming_writer import ArticleTooShortError, GamingArticleWriter
from src.publishers.blogger_client import BloggerClient
from src.publishers.oauth_helper import export_secrets as build_export_secrets
from src.publishers.oauth_helper import interactive_login, mask_secret

app = typer.Typer(
    help="Automated SEO publishing pipeline for Android Game Hack Area.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Restrict to a category id."),
    topic_id: Optional[str] = typer.Option(None, "--topic-id", "-t", help="Publish a specific topic id."),
    draft: Optional[bool] = typer.Option(
        None, "--draft/--live", help="Publish as draft or live. Defaults to DEFAULT_PUBLISH_STATUS."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Generate the article but skip publishing and history recording."
    ),
) -> None:
    """Run one publishing cycle: pick a topic, generate the article, publish it to Blogger."""
    settings = get_settings()
    catalog = load_catalog(TOPICS_FILE)
    history = HistoryDB(DB_PATH)

    try:
        category_id, topic = history.get_next_topic(catalog, category_id=category, topic_id=topic_id)
    except ValueError as exc:
        console.print(Panel(str(exc), title="No topic available", style="red"))
        raise typer.Exit(code=1)

    category_label = catalog.categories[category_id].label
    console.print(f"[bold cyan]Selected topic:[/] {topic.title} [dim]({category_id})[/]")

    with console.status("Generating article with Gemini..."):
        try:
            writer = GamingArticleWriter(settings)
            article = writer.generate_article(topic, category_label)
        except (RuntimeError, ArticleTooShortError) as exc:
            console.print(Panel(str(exc), title="Generation failed", style="red"))
            raise typer.Exit(code=1)

    console.print(
        f"[green]Generated[/] '{article.title}' — {article.word_count} words, "
        f"labels: {', '.join(article.labels)}"
    )

    is_draft = draft if draft is not None else settings.DEFAULT_PUBLISH_STATUS == "DRAFT"

    if dry_run:
        output_dir = DB_PATH.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = output_dir / f"{topic.id}.html"
        preview_path.write_text(f"<h1>{article.title}</h1>\n{article.html_content}", encoding="utf-8")
        console.print(Panel(f"Dry run only — nothing published. Preview saved to {preview_path}", style="yellow"))
        return

    try:
        blogger = BloggerClient(settings)
        result = blogger.publish_article(
            title=article.title,
            content=article.html_content,
            labels=article.labels,
            is_draft=is_draft,
        )
    except RuntimeError as exc:
        console.print(Panel(str(exc), title="Publish failed", style="red"))
        raise typer.Exit(code=1)

    history.record_publication(
        topic_id=topic.id,
        title=article.title,
        url=result.get("url"),
        blogger_post_id=result.get("id"),
        status="DRAFT" if is_draft else "LIVE",
        category=category_id,
        word_count=article.word_count,
    )

    status_label = "DRAFT" if is_draft else "LIVE"
    console.print(
        Panel(
            f"[bold]{article.title}[/]\nStatus: {status_label}\nURL: {result.get('url', 'n/a')}",
            title="Published",
            style="green",
        )
    )


@app.command()
def categories() -> None:
    """List all categories with topic and publication counts."""
    catalog = load_catalog(TOPICS_FILE)
    history = HistoryDB(DB_PATH)
    stats = history.get_stats(catalog)

    table = Table(title="Topic Categories")
    table.add_column("Category ID", style="cyan")
    table.add_column("Label")
    table.add_column("Total", justify="right")
    table.add_column("Published", justify="right", style="green")
    table.add_column("Remaining", justify="right", style="yellow")

    for cid, category in catalog.categories.items():
        s = stats[cid]
        table.add_row(cid, category.label, str(s["total"]), str(s["published"]), str(s["remaining"]))

    console.print(table)


@app.command()
def auth(force: bool = typer.Option(False, "--force", help="Force a fresh interactive login.")) -> None:
    """Validate stored Blogger credentials, or run an interactive login."""
    settings = get_settings()

    try:
        if force:
            console.print("Starting interactive OAuth login...")
            interactive_login(settings)
        blogger = BloggerClient(settings)
        info = blogger.get_blog_info()
    except RuntimeError as exc:
        console.print(Panel(str(exc), title="Authentication failed", style="red"))
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"Connected to [bold]{info.get('name')}[/]\n{info.get('url')}",
            title="Blogger auth OK",
            style="green",
        )
    )


@app.command()
def stats() -> None:
    """Show published vs remaining topic breakdown and recent posts."""
    catalog = load_catalog(TOPICS_FILE)
    history = HistoryDB(DB_PATH)
    category_stats = history.get_stats(catalog)

    total = sum(s["total"] for s in category_stats.values())
    published = sum(s["published"] for s in category_stats.values())

    console.print(f"[bold]{published}[/] / {total} topics published ({total - published} remaining)")

    table = Table(title="By Category")
    table.add_column("Category", style="cyan")
    table.add_column("Published", justify="right", style="green")
    table.add_column("Remaining", justify="right", style="yellow")
    for cid, s in category_stats.items():
        table.add_row(catalog.categories[cid].label, str(s["published"]), str(s["remaining"]))
    console.print(table)

    recent = history.list_recent(limit=10)
    if recent:
        recent_table = Table(title="Recently Published")
        recent_table.add_column("Title")
        recent_table.add_column("Status")
        recent_table.add_column("Published At")
        for post in recent:
            recent_table.add_row(post.title, post.status, post.published_at)
        console.print(recent_table)


@app.command(name="export-secrets")
def export_secrets(
    reveal: bool = typer.Option(False, "--reveal", help="Print full unmasked values (sensitive)."),
) -> None:
    """Print GitHub Actions secret values for this project's Blogger/Gemini credentials."""
    settings = get_settings()
    values = build_export_secrets(settings)

    if not values:
        console.print(Panel("No credential values found locally to export.", style="red"))
        raise typer.Exit(code=1)

    table = Table(title="GitHub Secrets")
    table.add_column("Secret Name", style="cyan")
    table.add_column("Value")
    for name, value in values.items():
        table.add_row(name, value if reveal else mask_secret(value))
    console.print(table)

    if not reveal:
        console.print("[dim]Run with --reveal to print full values. Never commit these.[/dim]")


if __name__ == "__main__":
    app()
