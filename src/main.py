from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import DB_PATH, TOPICS_FILE, get_settings
from src.catalog import load_catalog
from src.db.history import HistoryDB
from src.generators.an1_formatter import AN1Formatter, ContentEnhancementError
from src.generators.gaming_writer import ArticleTooShortError, GamingArticleWriter
from src.publishers.blogger_client import BloggerClient
from src.publishers.oauth_helper import export_secrets as build_export_secrets
from src.publishers.oauth_helper import interactive_login, mask_secret
from src.scrapers.an1_scraper import AN1Post, AN1Scraper, AN1ScraperError

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


@app.command(name="an1-post")
def an1_post(
    url: str = typer.Argument(..., help="AN1 post URL to scrape and publish."),
    draft: Optional[bool] = typer.Option(
        None, "--draft/--live", help="Publish as draft or live. Defaults to DEFAULT_PUBLISH_STATUS."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scrape and format without publishing."
    ),
) -> None:
    """Scrape a specific AN1 post, generate a rich blog page, and publish to Blogger."""
    settings = get_settings()
    history = HistoryDB(DB_PATH)
    scraper = AN1Scraper()
    formatter = AN1Formatter(settings=settings)

    with console.status(f"Scraping {url}..."):
        try:
            post = scraper.scrape_post(url, resolve_download=False)
            scraper.validate_post(post)
        except AN1ScraperError as exc:
            console.print(Panel(str(exc), title="Scrape or validation failed", style="red"))
            raise typer.Exit(code=1)

    if not dry_run and history.is_an1_published(post.post_id, post.version):
        console.print(Panel(f"Post {post.app_name} v{post.version} has already been published to Blogger.", title="Already Published", style="yellow"))
        return

    with console.status("Resolving direct download link..."):
        scraper.resolve_download_link(post)

    title = formatter.build_post_title(post)
    labels = formatter.build_labels(post)
    try:
        with console.status("Generating article content..."):
            html_content = formatter.format_html(post)
    except ContentEnhancementError as exc:
        console.print(Panel(str(exc), title="Content generation failed", style="red"))
        raise typer.Exit(code=1)

    console.print(
        f"[green]Scraped & Formatted[/] '[bold]{post.app_name}[/]' (v{post.version}) — "
        f"Primary Link: {post.dw_page_url or 'n/a'}"
    )

    is_draft = draft if draft is not None else settings.DEFAULT_PUBLISH_STATUS == "DRAFT"

    if dry_run:
        output_dir = DB_PATH.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = output_dir / f"an1_{post.post_id}.html"
        preview_path.write_text(f"<h1>{title}</h1>\n{html_content}", encoding="utf-8")
        console.print(Panel(f"Dry run only — preview saved to {preview_path}", style="yellow"))
        return

    blogger = BloggerClient(settings)
    existing = history.get_existing_blogger_post(post.post_id)

    if existing and existing.blogger_post_id:
        with console.status(f"Updating existing Blogger post to v{post.version}..."):
            try:
                result = blogger.update_article(
                    post_id=existing.blogger_post_id,
                    title=title,
                    content=html_content,
                    labels=labels,
                )
            except RuntimeError as exc:
                console.print(Panel(str(exc), title="Blogger update failed", style="red"))
                raise typer.Exit(code=1)
        action_name = "Updated"
    else:
        with console.status("Publishing to Blogger..."):
            try:
                result = blogger.publish_article(
                    title=title,
                    content=html_content,
                    labels=labels,
                    is_draft=is_draft,
                )
            except RuntimeError as exc:
                console.print(Panel(str(exc), title="Publish failed", style="red"))
                raise typer.Exit(code=1)
        action_name = "Published"

    history.record_an1_publication(
        post_id=post.post_id,
        version=post.version,
        source_url=post.url,
        title=title,
        direct_download_url=post.direct_download_url,
        dw_page_url=post.dw_page_url,
        blogger_post_id=result.get("id"),
        blogger_url=result.get("url"),
        status="DRAFT" if is_draft else "LIVE",
    )

    status_label = "DRAFT" if is_draft else "LIVE"
    console.print(
        Panel(
            f"[bold]{title}[/]\nAction: {action_name}\nStatus: {status_label}\nBlogger URL: {result.get('url', 'n/a')}\nPrimary DW Page: {post.dw_page_url}",
            title=f"AN1 Post {action_name}",
            style="green",
        )
    )


@app.command(name="an1-sync")
def an1_sync(
    limit: int = typer.Option(1, "--limit", "-l", help="Maximum number of new posts to publish in this run."),
    draft: Optional[bool] = typer.Option(
        None, "--draft/--live", help="Publish as draft or live. Defaults to DEFAULT_PUBLISH_STATUS."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Scrape and format without publishing."
    ),
) -> None:
    """Discover newly published posts on AN1.com and publish them to Blogger."""
    settings = get_settings()
    history = HistoryDB(DB_PATH)
    scraper = AN1Scraper()
    formatter = AN1Formatter(settings=settings)

    with console.status("Checking AN1.com for latest posts..."):
        discovered_urls = scraper.fetch_latest_post_urls(limit=40)

    if not discovered_urls:
        console.print(Panel("Could not discover posts from AN1.com.", style="yellow"))
        return

    # Process discovered URLs oldest-first so the backlog is consumed chronologically
    candidate_urls = list(reversed(discovered_urls))
    published_keys = history.get_published_an1_keys()

    console.print(f"[bold cyan]Scanning {len(candidate_urls)} posts from AN1.com...[/]")

    # Hoist BloggerClient so authentication happens once
    blogger: Optional[BloggerClient] = None
    if not dry_run:
        try:
            blogger = BloggerClient(settings)
        except RuntimeError as exc:
            console.print(Panel(str(exc), title="Blogger auth failed", style="red"))
            raise typer.Exit(code=1)

    published_count = 0
    failed_count = 0

    for url in candidate_urls:
        if published_count >= limit:
            break

        with console.status(f"Processing {url}..."):
            # Scraped without the download page so already-published posts cost one request.
            try:
                post = scraper.scrape_post(url, resolve_download=False)
                scraper.validate_post(post)
            except AN1ScraperError as exc:
                failed_count += 1
                console.print(f"[yellow]Skipping {url} (validation/scrape error):[/] {exc}")
                continue

            # Check if this exact (post_id, version) is already recorded
            if (post.post_id, post.version) in published_keys or history.is_an1_published(post.post_id, post.version):
                continue

            scraper.resolve_download_link(post)

            title = formatter.build_post_title(post)
            labels = formatter.build_labels(post)
            try:
                html_content = formatter.format_html(post)
            except ContentEnhancementError as exc:
                failed_count += 1
                console.print(f"[yellow]Skipping {title} (content generation failed):[/] {exc}")
                continue

            is_draft = draft if draft is not None else settings.DEFAULT_PUBLISH_STATUS == "DRAFT"

            if dry_run:
                output_dir = DB_PATH.parent.parent / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                preview_path = output_dir / f"an1_{post.post_id}.html"
                preview_path.write_text(f"<h1>{title}</h1>\n{html_content}", encoding="utf-8")
                console.print(f"[yellow]Dry-run saved:[/] {preview_path}")
                published_count += 1
                continue

            existing = history.get_existing_blogger_post(post.post_id)
            try:
                assert blogger is not None
                if existing and existing.blogger_post_id:
                    result = blogger.update_article(
                        post_id=existing.blogger_post_id,
                        title=title,
                        content=html_content,
                        labels=labels,
                    )
                    action = "Updated"
                else:
                    result = blogger.publish_article(
                        title=title,
                        content=html_content,
                        labels=labels,
                        is_draft=is_draft,
                    )
                    action = "Published"
            except RuntimeError as exc:
                console.print(f"[red]Failed to publish {title}:[/] {exc}")
                continue

            history.record_an1_publication(
                post_id=post.post_id,
                version=post.version,
                source_url=post.url,
                title=title,
                direct_download_url=post.direct_download_url,
                dw_page_url=post.dw_page_url,
                blogger_post_id=result.get("id"),
                blogger_url=result.get("url"),
                status="DRAFT" if is_draft else "LIVE",
            )
            published_count += 1
            console.print(f"[green]{action}:[/] {title} -> {result.get('url', 'n/a')}")

    # Nothing published *and* posts failed means the pipeline is broken (most likely an AN1
    # markup change), not that the backlog is simply empty. Fail the run so CI reports it
    # instead of showing a green "nothing new" forever.
    if failed_count and published_count == 0:
        console.print(
            Panel(
                f"{failed_count} of {len(candidate_urls)} discovered post(s) failed scraping, validation, "
                "or content generation, and nothing was published. AN1 markup may have changed.",
                title="Sync failed",
                style="red",
            )
        )
        raise typer.Exit(code=1)

    if published_count == 0:
        console.print(Panel("No new unpublished posts or version updates found on AN1.com.", style="green"))
    else:
        summary = f"Successfully processed {published_count} AN1 post(s)."
        if failed_count:
            summary += f" {failed_count} post(s) skipped after errors."
        console.print(Panel(summary, style="green"))


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
