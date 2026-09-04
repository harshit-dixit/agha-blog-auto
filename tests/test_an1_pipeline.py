from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.db.history import HistoryDB
from src.generators.an1_formatter import AN1Formatter, ContentEnhancementError
from src.main import app
from src.scrapers.an1_scraper import (
    AN1Post,
    AN1Scraper,
    AN1ValidationError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _formatter_with_fake_gemini(*, response_text: str = "", exc: Exception | None = None) -> AN1Formatter:
    """Build a formatter wired to a stubbed Gemini client (never touches the network)."""
    formatter = AN1Formatter()
    fake_settings = MagicMock()
    fake_settings.GEMINI_MODEL = "gemini-3.5-flash"
    formatter.settings = fake_settings

    client = MagicMock()
    if exc is not None:
        client.models.generate_content.side_effect = exc
    else:
        client.models.generate_content.return_value = MagicMock(text=response_text)
    formatter._genai_client = client
    return formatter


def _fixture_session_mocks() -> tuple[object, object]:
    """Return (mock_get, mock_head) callables serving the saved AN1 fixtures."""
    post_html = (FIXTURES_DIR / "sample_post.html").read_text(encoding="utf-8")
    dw_html = (FIXTURES_DIR / "sample_dw.html").read_text(encoding="utf-8")

    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = dw_html if "file_" in url else post_html
        return mock_resp

    def mock_head(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return mock_resp

    return mock_get, mock_head


@pytest.fixture
def sample_post() -> AN1Post:
    return AN1Post(
        post_id="4683",
        url="https://an1.com/4683-subway-surfers-mod-apk-8.html",
        title="Download Subway Surfers (MOD, Unlimited Coins/Keys) 3.68.5 free on android",
        app_name="Subway Surfers",
        icon_url="https://an1.com/uploads/posts/subway.png",
        developer="SYBO Games",
        categories=["Games", "Arcade"],
        version="3.68.5",
        android_version="Android 6.0+",
        size="252.1Mb",
        updated_date="September 2, 2026",
        rating="4.8",
        installs="1,000,000,000+",
        mod_features="MOD, Unlimited Coins/Keys",
        description_html="<p>Subway Surfers running game.</p>",
        description_text="Subway Surfers is the legendary running game on mobile. Run along tracks and dodge trains.",
        screenshots=["https://an1.com/screen1.webp", "https://an1.com/screen2.webp"],
        dw_page_url="https://an1.com/file_4683-dw.html",
        direct_download_url="https://files.an1.net/subway-surfers-mod_3.68.5-an1.com.apk",
    )


def test_extract_post_id():
    scraper = AN1Scraper(request_delay=0)
    assert scraper.extract_post_id("https://an1.com/4683-subway-surfers-mod-apk-8.html") == "4683"
    assert scraper.extract_post_id("https://an1.com/266-sniper-3d.html") == "266"
    assert scraper.extract_post_id("https://an1.com/custom-game.html") == "custom-game"


def test_mocked_scrape_post_with_fixtures():
    scraper = AN1Scraper(request_delay=0, verify_direct_link=True)
    mock_get, mock_head = _fixture_session_mocks()

    with patch.object(scraper.session, "get", side_effect=mock_get), \
         patch.object(scraper.session, "head", side_effect=mock_head):
        post = scraper.scrape_post("https://an1.com/4683-subway-surfers-mod-apk-8.html")

    assert post.post_id == "4683"
    assert post.app_name == "Subway Surfers"
    assert post.version == "3.68.5"
    assert post.size == "252.1Mb"
    assert post.dw_page_url == "https://an1.com/file_4683-dw.html"
    assert post.direct_download_url == "https://files.an1.net/subway-surfers-mod_3.68.5-an1.com.apk"

    # Must pass validation
    scraper.validate_post(post)


def test_validation_gate_fails_on_degraded_post():
    scraper = AN1Scraper(request_delay=0)

    degraded = AN1Post(
        post_id="999",
        url="https://an1.com/999.html",
        title="Unknown Game",
        app_name="Android App",
        icon_url="",
        developer="Unknown",
        categories=[],
        version="Latest",
        android_version="",
        size="Unknown",
        updated_date="",
        rating="",
        installs="",
        mod_features="",
        description_html="",
        description_text="",
        dw_page_url=None,
    )

    with pytest.raises(AN1ValidationError) as exc_info:
        scraper.validate_post(degraded)

    err = str(exc_info.value)
    assert "Invalid or sentinel app_name" in err
    assert "Invalid or sentinel version" in err
    assert "Invalid or missing dw_page_url" in err


def test_formatter_inverts_buttons_and_removes_unverified_claims(sample_post: AN1Post):
    formatter = AN1Formatter()
    html_content = formatter.format_html(sample_post)
    title = formatter.build_post_title(sample_post)
    labels = formatter.build_labels(sample_post)

    assert "Subway Surfers" in title
    assert "MOD APK" in labels

    # Primary button must link to the stable AN1 download page with AGHA theme class
    assert f'href="{sample_post.dw_page_url}" class="dl-btn"' in html_content

    # Secondary mirror must link to the direct APK with AGHA pink alt button class
    assert f'href="{sample_post.direct_download_url}" class="dl-btn alt"' in html_content

    # Unverified claims must be removed
    assert "Verified Clean • No Malware" not in html_content
    assert "verified to be virus-free" not in html_content


def test_formatter_conforms_to_agha_clay_design_system(sample_post: AN1Post):
    """Verify that generated post HTML matches the AGHA Theme Design System."""
    formatter = AN1Formatter()
    html_content = formatter.format_html(sample_post)

    # Theme native component classes
    assert 'class="app-head"' in html_content
    assert 'class="app-box"' in html_content
    assert 'class="app-row"' in html_content
    assert 'class="screens"' in html_content
    assert 'class="steps"' in html_content
    assert 'class="faq"' in html_content
    assert 'class="notice"' in html_content

    # AGHA Theme saturated pills following contrast rules
    assert 'class="pill pill-pink"' in html_content
    assert 'class="pill pill-teal"' in html_content
    assert 'class="pill pill-ochre"' in html_content

    # AGHA Design tokens present
    assert "--canvas: #fffaf0" in html_content
    assert "--card: #f5f0e0" in html_content
    assert "--primary: #0a0a0a" in html_content
    assert "--pink: #ff4d8b" in html_content
    assert "--teal: #1a3a3a" in html_content
    assert "--lav: #b8a4ed" in html_content
    assert "--ochre: #e8b94a" in html_content

    # Disallowed dark slate and neon green themes must not be present
    assert "#0f172a" not in html_content
    assert "#1e293b" not in html_content
    assert "#10b981" not in html_content
    assert "badge-green" not in html_content
    assert "badge-blue" not in html_content


def test_history_db_version_tracking_and_corrupt_ledger_handling(tmp_path: Path, sample_post: AN1Post):
    db_path = tmp_path / "test_history.db"
    json_path = tmp_path / "an1_published.json"
    history = HistoryDB(db_path, json_tracker_path=json_path)

    # Initially unpublished
    assert not history.is_an1_published(sample_post.post_id, sample_post.version)

    # Record version 1
    history.record_an1_publication(
        post_id=sample_post.post_id,
        version="3.68.5",
        source_url=sample_post.url,
        title=sample_post.title,
        direct_download_url=sample_post.direct_download_url,
        dw_page_url=sample_post.dw_page_url,
        blogger_post_id="post-111",
        blogger_url="https://example.blogspot.com/post-v1.html",
        status="LIVE",
    )

    # v3.68.5 is published, but v3.69.0 is NOT
    assert history.is_an1_published(sample_post.post_id, "3.68.5")
    assert not history.is_an1_published(sample_post.post_id, "3.69.0")

    # Lookup existing Blogger post for version bumps
    existing = history.get_existing_blogger_post(sample_post.post_id)
    assert existing is not None
    assert existing.blogger_post_id == "post-111"

    # Test corrupt JSON raises RuntimeError
    json_path.write_text("{corrupt json", encoding="utf-8")
    corrupt_db_path = tmp_path / "corrupt_history.db"
    with pytest.raises(RuntimeError) as exc_info:
        HistoryDB(corrupt_db_path, json_tracker_path=json_path)
    assert "Corrupt or unreadable AN1 publication ledger" in str(exc_info.value)


def test_cli_an1_post_dry_run_offline(tmp_path: Path, sample_post: AN1Post):
    runner = CliRunner()

    # AN1Formatter is replaced with a settings-less instance so the run cannot reach Gemini
    # even when GEMINI_API_KEY is present in the environment.
    with patch("src.main.AN1Scraper.scrape_post", return_value=sample_post), \
         patch("src.main.AN1Scraper.validate_post", return_value=None), \
         patch("src.main.AN1Formatter", side_effect=lambda **kwargs: AN1Formatter()), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(
            app,
            ["an1-post", sample_post.url, "--dry-run"],
        )

    assert result.exit_code == 0
    assert "Scraped & Formatted" in result.output
    assert "Subway Surfers" in result.output
    assert "Primary Link" in result.output


def test_scrape_post_defers_download_resolution():
    """resolve_download=False must not fetch the dw page or run the HEAD check."""
    scraper = AN1Scraper(request_delay=0, verify_direct_link=True)
    mock_get, mock_head = _fixture_session_mocks()

    with patch.object(scraper.session, "get", side_effect=mock_get) as get_spy, \
         patch.object(scraper.session, "head", side_effect=mock_head) as head_spy:
        post = scraper.scrape_post(
            "https://an1.com/4683-subway-surfers-mod-apk-8.html",
            resolve_download=False,
        )

        # Only the post page itself was fetched
        assert get_spy.call_count == 1
        assert head_spy.call_count == 0
        assert post.direct_download_url is None
        # dw_page_url is still parsed from the post page, so validation can run
        assert post.dw_page_url == "https://an1.com/file_4683-dw.html"
        scraper.validate_post(post)

        # Resolving afterwards fills in the direct link
        scraper.resolve_download_link(post)
        assert get_spy.call_count == 2
        assert post.direct_download_url == "https://files.an1.net/subway-surfers-mod_3.68.5-an1.com.apk"


def test_resolve_download_link_is_noop_when_already_resolved(sample_post: AN1Post):
    scraper = AN1Scraper(request_delay=0)
    with patch.object(scraper.session, "get") as get_spy:
        scraper.resolve_download_link(sample_post)
    assert get_spy.call_count == 0


def test_validation_rejects_unparsed_version(sample_post: AN1Post):
    """An empty version must fail the gate rather than publish under a placeholder."""
    scraper = AN1Scraper(request_delay=0)
    sample_post.version = ""
    with pytest.raises(AN1ValidationError, match="version"):
        scraper.validate_post(sample_post)


def test_size_dropped_from_button_label_when_unknown(sample_post: AN1Post):
    sample_post.size = "Unknown"
    html_content = AN1Formatter().format_html(sample_post)

    assert "Download APK (Unknown)" not in html_content
    assert "File Size Unknown" not in html_content
    assert "Download APK</span>" in html_content

    # A known size is still shown
    sample_post.size = "252.1Mb"
    assert "Download APK (252.1Mb)" in AN1Formatter().format_html(sample_post)


def test_gemini_output_is_stripped_of_fences_and_preamble(sample_post: AN1Post):
    body = "<p>" + ("Original review prose. " * 20) + "</p>"
    formatter = _formatter_with_fake_gemini(
        response_text=f"Sure! Here is the review:\n```html\n{body}\n```"
    )

    html_content = formatter.format_html(sample_post)

    assert "Here is the review" not in html_content
    assert "```" not in html_content
    assert "Original review prose." in html_content


def test_gemini_api_failure_raises_instead_of_publishing_scraped_text(sample_post: AN1Post):
    formatter = _formatter_with_fake_gemini(exc=RuntimeError("429 RESOURCE_EXHAUSTED"))

    with pytest.raises(ContentEnhancementError, match="429"):
        formatter.format_html(sample_post)


def test_unusable_gemini_output_raises(sample_post: AN1Post):
    formatter = _formatter_with_fake_gemini(response_text="I cannot help with that request.")

    with pytest.raises(ContentEnhancementError, match="unusable"):
        formatter.format_html(sample_post)


def test_sync_exits_nonzero_when_every_post_fails(tmp_path: Path):
    """A markup change that breaks every post must fail the run, not report a green no-op."""
    runner = CliRunner()
    urls = [
        "https://an1.com/4683-subway-surfers-mod-apk-8.html",
        "https://an1.com/266-sniper-3d.html",
    ]

    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=urls), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=AN1ValidationError("no title found")), \
         patch("src.main.AN1Formatter", side_effect=lambda **kwargs: AN1Formatter()), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run", "--limit", "2"])

    assert result.exit_code == 1
    assert "Sync failed" in result.output


def test_sync_stays_green_when_backlog_is_simply_empty(tmp_path: Path, sample_post: AN1Post):
    """No failures and nothing new is a success, not an error."""
    runner = CliRunner()
    db_path = tmp_path / "history.db"
    json_path = tmp_path / "an1_published.json"

    history = HistoryDB(db_path, json_tracker_path=json_path)
    history.record_an1_publication(
        post_id=sample_post.post_id,
        version=sample_post.version,
        source_url=sample_post.url,
        title=sample_post.title,
        dw_page_url=sample_post.dw_page_url,
        blogger_post_id="post-111",
        status="LIVE",
    )

    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=[sample_post.url]), \
         patch("src.main.AN1Scraper.scrape_post", return_value=sample_post), \
         patch("src.main.AN1Scraper.validate_post", return_value=None), \
         patch("src.main.AN1Formatter", side_effect=lambda **kwargs: AN1Formatter()), \
         patch("src.main.DB_PATH", db_path):
        result = runner.invoke(app, ["an1-sync", "--dry-run", "--limit", "1"])

    assert result.exit_code == 0
    assert "No new unpublished posts" in result.output


def test_fetch_latest_post_urls_defaults_to_tags_mods():
    scraper = AN1Scraper(request_delay=0)
    page_html = """
    <html><body>
    <div class="data"><div class="name"><a href="/1001-mod-game-1.html">Game 1 (MOD)</a></div></div>
    <div class="data"><div class="name"><a href="/1002-mod-game-2.html">Game 2 (MOD)</a></div></div>
    </body></html>
    """
    mock_resp = MagicMock(status_code=200, text=page_html)

    with patch.object(scraper.session, "get", return_value=mock_resp) as mock_get:
        urls = scraper.fetch_latest_post_urls(limit=10)

    assert mock_get.call_count == 1
    assert mock_get.call_args[0][0] == "https://an1.com/tags/mods/"
    assert urls == [
        "https://an1.com/1001-mod-game-1.html",
        "https://an1.com/1002-mod-game-2.html",
    ]


def test_fetch_latest_post_urls_paginates_when_limit_exceeds_single_page():
    scraper = AN1Scraper(request_delay=0)
    page1_html = """
    <html><body>
    <div class="data"><div class="name"><a href="/1001-mod-game-1.html">Game 1 (MOD)</a></div></div>
    <div class="data"><div class="name"><a href="/1002-mod-game-2.html">Game 2 (MOD)</a></div></div>
    <div class="nav_more"><a href="https://an1.com/tags/mods/page/2/">More...</a></div>
    </body></html>
    """
    page2_html = """
    <html><body>
    <div class="data"><div class="name"><a href="/1003-mod-game-3.html">Game 3 (MOD)</a></div></div>
    <div class="data"><div class="name"><a href="/1004-mod-game-4.html">Game 4 (MOD)</a></div></div>
    </body></html>
    """

    def mock_get(url, *args, **kwargs):
        resp = MagicMock(status_code=200)
        resp.text = page2_html if "page/2" in url else page1_html
        return resp

    with patch.object(scraper.session, "get", side_effect=mock_get) as mock_get_spy:
        urls = scraper.fetch_latest_post_urls(limit=3)

    assert mock_get_spy.call_count == 2
    assert len(urls) == 3
    assert urls == [
        "https://an1.com/1001-mod-game-1.html",
        "https://an1.com/1002-mod-game-2.html",
        "https://an1.com/1003-mod-game-3.html",
    ]


def test_fetch_latest_post_urls_honors_custom_sources():
    scraper = AN1Scraper(request_delay=0)
    page_html = """
    <html><body>
    <div class="data"><div class="name"><a href="/2001-custom-game.html">Custom Game</a></div></div>
    </body></html>
    """
    mock_resp = MagicMock(status_code=200, text=page_html)

    with patch.object(scraper.session, "get", return_value=mock_resp) as mock_get:
        urls = scraper.fetch_latest_post_urls(limit=5, sources=["https://an1.com/games/"])

    assert mock_get.call_count == 1
    assert mock_get.call_args[0][0] == "https://an1.com/games/"
    assert urls == ["https://an1.com/2001-custom-game.html"]


def test_cli_an1_sync_passes_custom_source(tmp_path: Path):
    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=[]) as mock_fetch, \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run", "--source", "https://an1.com/custom-tag/"])

    assert result.exit_code == 0
    mock_fetch.assert_called_once_with(limit=40, sources=["https://an1.com/custom-tag/"])

