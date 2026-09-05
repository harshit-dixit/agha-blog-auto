from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from typer.testing import CliRunner

from src.db.history import HistoryDB
from src.generators.an1_formatter import (
    AN1Formatter,
    ContentEnhancementError,
    QuotaExhaustedError,
)
from src.main import app
from src.publishers.blogger_client import BloggerRateLimitError
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
    fake_settings.GEMINI_MODEL = "gemini-3.5-flash-lite"
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


def test_cli_an1_sync_defaults_to_limit_10_and_handles_fewer_new_posts(tmp_path: Path):
    db_path = tmp_path / "history.db"
    history = HistoryDB(db_path)
    # Pre-record post 1001 and 1002 as already published
    history.record_an1_publication(
        post_id="1001",
        version="1.0",
        source_url="https://an1.com/1001-game-1.html",
        title="Game 1 Mod",
        direct_download_url="https://an1.com/dl1",
        dw_page_url="https://an1.com/dw1",
        blogger_post_id="b1",
        blogger_url="https://blog.com/1",
        status="LIVE",
    )
    history.record_an1_publication(
        post_id="1002",
        version="1.0",
        source_url="https://an1.com/1002-game-2.html",
        title="Game 2 Mod",
        direct_download_url="https://an1.com/dl2",
        dw_page_url="https://an1.com/dw2",
        blogger_post_id="b2",
        blogger_url="https://blog.com/2",
        status="LIVE",
    )

    discovered = [
        "https://an1.com/1001-game-1.html",
        "https://an1.com/1002-game-2.html",
        "https://an1.com/1003-game-3.html",
    ]

    def mock_scrape(url, resolve_download=False):
        pid = url.split("/")[-1].split("-")[0]
        return AN1Post(
            post_id=pid,
            url=url,
            title=f"Game {pid} (MOD, Unlimited Coins)",
            app_name=f"Game {pid}",
            icon_url="https://an1.com/img.png",
            developer="Dev",
            categories=["Action"],
            version="1.0",
            android_version="5.0 and up",
            size="100 Mb",
            updated_date="Today",
            rating="4.5",
            installs="10,000+",
            mod_features="Unlimited Coins",
            description_html="<p>Fun action game.</p>",
            description_text="Fun action game.",
            dw_page_url=f"https://an1.com/dw_{pid}.html",
        )

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered) as mock_fetch, \
         patch("src.main.AN1Scraper.scrape_post", side_effect=mock_scrape), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=False), \
         patch("src.main.AN1Formatter.format_html", return_value="<p>Review content</p>"), \
         patch("src.main.DB_PATH", db_path):
        result = runner.invoke(app, ["an1-sync", "--dry-run"])

    assert result.exit_code == 0
    # Verified that default limit is 10 and dynamic discovery buffer was used
    mock_fetch.assert_called_once_with(limit=40, sources=None)
    # Output should show that only 1 post was processed (since 2 were already in history)
    assert "Successfully processed 1 AN1 post(s)" in result.stdout


def _batch_posts(count: int) -> list[AN1Post]:
    """Build `count` minimal, distinct posts for batch-generation tests."""
    return [
        AN1Post(
            post_id=str(2000 + i),
            url=f"https://an1.com/{2000 + i}-game-{i}.html",
            title=f"Game {i} (MOD, Unlimited Coins)",
            app_name=f"Game {i}",
            icon_url="https://an1.com/img.png",
            developer="Dev",
            categories=["Action"],
            version="1.0",
            android_version="5.0 and up",
            size="100 Mb",
            updated_date="Today",
            rating="4.5",
            installs="10,000+",
            mod_features="Unlimited Coins",
            description_html="<p>Fun action game.</p>",
            description_text="Fun action game.",
            dw_page_url=f"https://an1.com/dw_{2000 + i}.html",
        )
        for i in range(count)
    ]


def _review(marker: str) -> str:
    return "<p>" + (f"{marker} review prose. " * 20) + "</p>"


def test_quota_error_is_flagged_as_run_level(sample_post: AN1Post):
    """A 429 must be distinguishable from an ordinary generation failure."""
    formatter = _formatter_with_fake_gemini(exc=RuntimeError("429 RESOURCE_EXHAUSTED. quota"))

    with pytest.raises(QuotaExhaustedError):
        formatter.format_html(sample_post)


def test_non_quota_error_stays_a_plain_enhancement_error(sample_post: AN1Post):
    formatter = _formatter_with_fake_gemini(exc=RuntimeError("500 INTERNAL"))

    with pytest.raises(ContentEnhancementError) as excinfo:
        formatter.format_html(sample_post)
    assert not isinstance(excinfo.value, QuotaExhaustedError)


def test_enhance_batch_maps_reviews_back_to_post_ids():
    posts = _batch_posts(3)
    payload = json.dumps(
        [{"post_id": p.post_id, "review_html": _review(p.app_name)} for p in posts]
    )
    formatter = _formatter_with_fake_gemini(response_text=payload)

    result = formatter.enhance_batch(posts)

    assert set(result) == {p.post_id for p in posts}
    assert "Game 1 review prose." in result["2001"]
    # One call covers the whole batch; that is the entire point of batching.
    assert formatter._genai_client.models.generate_content.call_count == 1


def test_enhance_batch_ignores_unknown_and_unusable_entries():
    posts = _batch_posts(2)
    payload = json.dumps(
        [
            {"post_id": "2000", "review_html": _review("Game 0")},
            {"post_id": "9999", "review_html": _review("Not requested")},
            {"post_id": "2001", "review_html": "too short"},
        ]
    )
    formatter = _formatter_with_fake_gemini(response_text=payload)

    result = formatter.enhance_batch(posts)

    # 2001 came back unusable and 9999 was never asked for, so only 2000 survives.
    assert set(result) == {"2000"}


def test_enhance_batch_salvages_whole_entries_from_a_truncated_reply():
    posts = _batch_posts(3)
    full = json.dumps(
        [{"post_id": p.post_id, "review_html": _review(p.app_name)} for p in posts]
    )
    # Simulate the reply being cut off partway through the third entry.
    truncated = full[: full.rindex("{")] + '{"post_id": "2002", "review_html": "<p>Game 2 rev'
    formatter = _formatter_with_fake_gemini(response_text=truncated)

    result = formatter.enhance_batch(posts)

    # The two complete entries are recovered; only the mangled one is lost.
    assert set(result) == {"2000", "2001"}


def test_enhance_batch_raises_quota_exhausted_on_429():
    formatter = _formatter_with_fake_gemini(exc=RuntimeError("429 RESOURCE_EXHAUSTED"))

    with pytest.raises(QuotaExhaustedError):
        formatter.enhance_batch(_batch_posts(2))


def test_enhance_batch_raises_when_nothing_usable_comes_back():
    formatter = _formatter_with_fake_gemini(response_text="I cannot help with that request.")

    with pytest.raises(ContentEnhancementError, match="no usable reviews"):
        formatter.enhance_batch(_batch_posts(2))


def test_rejected_batch_logs_the_shape_of_the_reply(caplog):
    """A rejected batch must name its own cause.

    Wrong field names, ids matching nothing and prose that is not <p> markup all surface
    identically as "no usable reviews", so the reply's actual shape has to reach the log or
    the next failure is undiagnosable after the fact.
    """
    reply = json.dumps([{"id": "2000", "review": _review("Right prose, wrong keys")}])
    formatter = _formatter_with_fake_gemini(response_text=reply)

    with caplog.at_level(logging.WARNING), pytest.raises(ContentEnhancementError):
        formatter.enhance_batch(_batch_posts(1))

    assert "Rejected batch reply" in caplog.text
    assert "1 object(s) parsed" in caplog.text
    # The keys it actually used - the fact that separates a renamed field from bad prose.
    assert "keys=['id', 'review']" in caplog.text
    assert "Right prose, wrong keys review prose." in caplog.text


def test_format_html_prefers_supplied_prose_over_a_fresh_call(sample_post: AN1Post):
    formatter = _formatter_with_fake_gemini(response_text=_review("Should not be used"))

    html_content = formatter.format_html(sample_post, enhanced_desc=_review("Batched"))

    assert "Batched review prose." in html_content
    assert "Should not be used" not in html_content
    formatter._genai_client.models.generate_content.assert_not_called()


def _sync_scrape_stub(url, resolve_download=False):
    pid = url.split("/")[-1].split("-")[0]
    return AN1Post(
        post_id=pid,
        url=url,
        title=f"Game {pid} (MOD, Unlimited Coins)",
        app_name=f"Game {pid}",
        icon_url="https://an1.com/img.png",
        developer="Dev",
        categories=["Action"],
        version="1.0",
        android_version="5.0 and up",
        size="100 Mb",
        updated_date="Today",
        rating="4.5",
        installs="10,000+",
        mod_features="Unlimited Coins",
        description_html="<p>Fun action game.</p>",
        description_text="Fun action game.",
        dw_page_url=f"https://an1.com/dw_{pid}.html",
    )


def test_sync_stops_scraping_once_the_limit_is_filled(tmp_path: Path):
    """A 40-URL discovery window must not cost 40 scrapes to publish 2 posts."""
    discovered = [f"https://an1.com/{3000 + i}-game-{i}.html" for i in range(40)]
    scraped: list[str] = []

    def counting_scrape(url, resolve_download=False):
        scraped.append(url)
        return _sync_scrape_stub(url, resolve_download)

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=counting_scrape), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=False), \
         patch("src.main.AN1Formatter.format_html", return_value="<p>Review content</p>"), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run", "--limit", "2"])

    assert result.exit_code == 0
    assert len(scraped) == 2, f"scraped {len(scraped)} posts to publish 2"


def test_sync_batches_generation_into_one_call_per_batch_size(tmp_path: Path):
    discovered = [f"https://an1.com/{4000 + i}-game-{i}.html" for i in range(6)]
    batch_sizes: list[int] = []

    def fake_batch(self, posts):
        batch_sizes.append(len(posts))
        return {p.post_id: _review(p.app_name) for p in posts}

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=True), \
         patch("src.main.AN1Formatter.enhance_batch", autospec=True, side_effect=fake_batch), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run", "--limit", "6"])

    assert result.exit_code == 0
    # Six posts at the default batch size of 2 means three calls, not six.
    assert batch_sizes == [2, 2, 2]
    assert "Successfully processed 6 AN1 post(s)" in result.stdout


def test_sync_defers_backlog_and_exits_zero_when_quota_is_exhausted(tmp_path: Path):
    """A spent quota is a deferral, not a broken pipeline, and must not publish raw text."""
    db_path = tmp_path / "history.db"
    discovered = [f"https://an1.com/{5000 + i}-game-{i}.html" for i in range(3)]

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=True), \
         patch(
             "src.main.AN1Formatter.enhance_batch",
             side_effect=QuotaExhaustedError("429 RESOURCE_EXHAUSTED"),
         ), \
         patch("src.main.AN1Formatter.format_html") as mock_format, \
         patch("src.main.DB_PATH", db_path):
        result = runner.invoke(app, ["an1-sync", "--dry-run"])

    assert result.exit_code == 0, result.stdout
    # Nothing was formatted, so nothing could have gone out with AN1's scraped text.
    mock_format.assert_not_called()
    assert "backlog" in result.stdout
    # The ledger is untouched, so the next run retries these posts.
    assert HistoryDB(db_path).get_published_an1_keys() == set()


def test_sync_stops_generating_after_the_first_quota_error(tmp_path: Path):
    """The first 429 must end generation instead of burning a request per remaining post."""
    discovered = [f"https://an1.com/{6000 + i}-game-{i}.html" for i in range(10)]
    calls = {"n": 0}

    def fail_with_quota(self, posts):
        calls["n"] += 1
        raise QuotaExhaustedError("429 RESOURCE_EXHAUSTED")

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=True), \
         patch("src.main.AN1Formatter.enhance_batch", autospec=True, side_effect=fail_with_quota), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run"])

    assert result.exit_code == 0
    # Ten posts would be two batches; the run stops after the first failure.
    assert calls["n"] == 1


def test_sync_publishes_the_batch_that_succeeded_before_the_quota_ran_out(tmp_path: Path):
    discovered = [f"https://an1.com/{7000 + i}-game-{i}.html" for i in range(10)]

    seen = {"n": 0}

    def first_batch_then_quota(self, posts):
        seen["n"] += 1
        if seen["n"] == 1:
            return {p.post_id: _review(p.app_name) for p in posts}
        raise QuotaExhaustedError("429 RESOURCE_EXHAUSTED")

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=True), \
         patch("src.main.AN1Formatter.enhance_batch", autospec=True, side_effect=first_batch_then_quota), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--dry-run"])

    assert result.exit_code == 0
    # Only the first batch got through before the quota error stopped generation.
    assert "Successfully processed 2 AN1 post(s)" in result.stdout


def test_json_ledger_keeps_every_row_past_the_list_default_limit(tmp_path: Path):
    """A ledger larger than list_an1_posts()' default page must survive a rewrite.

    In CI the SQLite file is disposable and rebuilt from this JSON each run, so a row
    dropped here is a post that gets republished as a duplicate on the next run.
    """
    db_path = tmp_path / "history.db"
    json_path = tmp_path / "an1_published.json"

    seeded = [
        {
            "post_id": str(i),
            "version": "1.0",
            "source_url": f"https://an1.com/{i}-app.html",
            "title": f"App {i}",
            "direct_download_url": None,
            "dw_page_url": f"https://an1.com/file_{i}-dw.html",
            "blogger_post_id": f"blogger-{i}",
            "blogger_url": f"https://example.blogspot.com/{i}.html",
            # Ascending timestamps, so post_id "0" is the oldest and the first row a
            # truncating "ORDER BY published_at DESC LIMIT 500" would discard.
            "published_at": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
            "status": "LIVE",
        }
        for i in range(600)
    ]
    json_path.write_text(json.dumps(seeded), encoding="utf-8")

    history = HistoryDB(db_path, json_tracker_path=json_path)
    assert len(history.list_an1_posts(limit=None)) == 600

    # Any publication rewrites the whole tracker file.
    history.record_an1_publication(
        post_id="new-post",
        version="2.0",
        source_url="https://an1.com/999-new.html",
        title="Newly Published App",
        dw_page_url="https://an1.com/file_999-dw.html",
        blogger_post_id="blogger-999",
        status="LIVE",
    )

    rewritten = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rewritten) == 601
    assert {"0", "1", "599", "new-post"} <= {item["post_id"] for item in rewritten}

    # A fresh run rebuilding SQLite from that JSON still knows the oldest posts.
    rebuilt = HistoryDB(tmp_path / "rebuilt.db", json_tracker_path=json_path)
    assert rebuilt.is_an1_published("0", "1.0")
    assert ("0", "1.0") in rebuilt.get_published_an1_keys()


# ---------------------------------------------------------------------------
# Blogger rate-limit handling and the generated-prose cache
# ---------------------------------------------------------------------------


def _http_error(status: int, reason: str = "rateLimitExceeded"):
    """Build an HttpError shaped like the ones Blogger returns."""
    from googleapiclient.errors import HttpError

    payload = {
        "error": {
            "code": status,
            "message": "Resource has been exhausted.",
            "errors": [{"message": "Resource has been exhausted.", "reason": reason}],
        }
    }
    return HttpError(MagicMock(status=status, reason=reason), json.dumps(payload).encode())


def _blogger_client(service: MagicMock, **overrides):
    """Build a BloggerClient with auth and API discovery stubbed out."""
    from config.settings import Settings
    from src.publishers.blogger_client import BloggerClient

    settings = Settings(BLOGGER_BLOG_ID="blog-1", **overrides)
    with patch("src.publishers.blogger_client.get_credentials", return_value=MagicMock()), \
         patch("src.publishers.blogger_client.build", return_value=service):
        return BloggerClient(settings)


def test_blogger_write_asks_googleapiclient_to_retry():
    """A burst 429 must be retried with backoff rather than losing the post outright."""
    service = MagicMock()
    insert = service.posts.return_value.insert.return_value
    insert.execute.return_value = {"id": "1", "url": "https://blog/1"}

    client = _blogger_client(service, BLOGGER_MAX_RETRIES=5, BLOGGER_MIN_WRITE_INTERVAL=0)
    client.publish_article(title="T", content="<p>c</p>", labels=[], is_draft=False)

    # num_retries is what makes googleapiclient back off on 429/5xx.
    assert insert.execute.call_args.kwargs["num_retries"] == 5


def test_blogger_429_raises_rate_limit_error_not_a_plain_failure():
    service = MagicMock()
    service.posts.return_value.insert.return_value.execute.side_effect = _http_error(429)

    client = _blogger_client(service, BLOGGER_MIN_WRITE_INTERVAL=0)
    with pytest.raises(BloggerRateLimitError):
        client.publish_article(title="T", content="<p>c</p>", labels=[], is_draft=False)


def test_blogger_403_permission_error_is_not_treated_as_a_rate_limit():
    """A genuine permission failure must stay a hard error instead of pausing the run."""
    service = MagicMock()
    service.posts.return_value.insert.return_value.execute.side_effect = _http_error(
        403, reason="forbidden"
    )

    client = _blogger_client(service, BLOGGER_MIN_WRITE_INTERVAL=0)
    with pytest.raises(RuntimeError) as exc_info:
        client.publish_article(title="T", content="<p>c</p>", labels=[], is_draft=False)
    assert not isinstance(exc_info.value, BloggerRateLimitError)


def test_blogger_paces_consecutive_writes():
    """Back-to-back inserts are what trip the burst limit, so writes must be spaced out."""
    service = MagicMock()
    service.posts.return_value.insert.return_value.execute.return_value = {"id": "1"}
    slept: list[float] = []

    client = _blogger_client(service, BLOGGER_MIN_WRITE_INTERVAL=3.0)
    with patch("src.publishers.blogger_client.time.sleep", side_effect=slept.append):
        client.publish_article(title="A", content="<p>a</p>", labels=[], is_draft=False)
        client.publish_article(title="B", content="<p>b</p>", labels=[], is_draft=False)

    # The first write goes out immediately; the second waits out the interval.
    assert len(slept) == 1
    assert 0 < slept[0] <= 3.0


def test_sync_stops_publishing_after_a_blogger_rate_limit(tmp_path: Path):
    """One 429 must end the run, not be re-tried once per remaining post."""
    discovered = [f"https://an1.com/{8000 + i}-game-{i}.html" for i in range(10)]
    attempts = {"n": 0}

    def publish(**kwargs):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            return {"id": str(attempts["n"]), "url": "https://blog/x"}
        raise BloggerRateLimitError("Blogger publish failed: 429 rateLimitExceeded")

    blogger = MagicMock()
    blogger.publish_article.side_effect = publish

    runner = CliRunner()
    with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
         patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
         patch("src.main.AN1Scraper.validate_post"), \
         patch("src.main.AN1Scraper.resolve_download_link"), \
         patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=False), \
         patch("src.main.AN1Formatter.format_html", return_value="<p>Review content</p>"), \
         patch("src.main.BloggerClient", return_value=blogger), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(app, ["an1-sync", "--limit", "10"])

    # Two published, the third hit the limit and ended the run: 10 eligible, 3 attempts.
    assert result.exit_code == 0, result.stdout
    assert attempts["n"] == 3, f"kept publishing after the rate limit ({attempts['n']} attempts)"
    assert "rate limit" in result.stdout.lower()
    assert "Successfully processed 2 AN1 post(s)" in result.stdout


def test_prose_survives_a_failed_publish_and_is_not_regenerated(tmp_path: Path):
    """The core waste: a Blogger failure must not cost a second Gemini request next run."""
    db_path = tmp_path / "history.db"
    discovered = [f"https://an1.com/{9000 + i}-game-{i}.html" for i in range(3)]
    batch_calls: list[list[str]] = []

    def fake_batch(self, posts):
        batch_calls.append([p.post_id for p in posts])
        return {p.post_id: _review(p.app_name) for p in posts}

    def run_sync(blogger):
        runner = CliRunner()
        with patch("src.main.AN1Scraper.fetch_latest_post_urls", return_value=discovered), \
             patch("src.main.AN1Scraper.scrape_post", side_effect=_sync_scrape_stub), \
             patch("src.main.AN1Scraper.validate_post"), \
             patch("src.main.AN1Scraper.resolve_download_link"), \
             patch("src.main.AN1Formatter.gemini_enabled", new_callable=PropertyMock, return_value=True), \
             patch("src.main.AN1Formatter.enhance_batch", autospec=True, side_effect=fake_batch), \
             patch("src.main.BloggerClient", return_value=blogger), \
             patch("src.main.DB_PATH", db_path):
            return runner.invoke(app, ["an1-sync", "--limit", "3"])

    # Run 1: Gemini generates all three, Blogger rate-limits the first insert.
    rate_limited = MagicMock()
    rate_limited.publish_article.side_effect = BloggerRateLimitError("429 rateLimitExceeded")
    first = run_sync(rate_limited)
    assert first.exit_code == 0, first.stdout
    # Pinning the exact count would only re-encode GEMINI_BATCH_SIZE; what matters is that
    # run 2 adds nothing to it.
    calls_after_run_1 = len(batch_calls)
    assert calls_after_run_1 > 0, "run 1 should have generated prose"

    # Run 2: Blogger recovers. The prose is already cached, so Gemini is never called.
    working = MagicMock()
    working.publish_article.side_effect = lambda **kw: {"id": "x", "url": "https://blog/x"}
    second = run_sync(working)

    assert second.exit_code == 0, second.stdout
    assert len(batch_calls) == calls_after_run_1, f"regenerated prose that was already cached: {batch_calls}"
    assert "Reused cached reviews for 3 post(s)" in second.stdout
    assert "Successfully processed 3 AN1 post(s)" in second.stdout

    # Published posts release their cache entries, so the cache tracks only the backlog.
    assert json.loads((tmp_path / "an1_prose_cache.json").read_text(encoding="utf-8")) == []


def test_prose_cache_survives_a_rebuilt_database(tmp_path: Path):
    """history.db is gitignored and rebuilt each CI run, so the cache must live in JSON."""
    db = HistoryDB(tmp_path / "history.db")
    db.cache_prose_many([("123", "2.0", "<p>cached prose</p>")])

    # A fresh DB in the same directory, as CI gets after checking out the ledger branch.
    (tmp_path / "history.db").unlink()
    rebuilt = HistoryDB(tmp_path / "history.db")

    assert rebuilt.get_cached_prose("123", "2.0") == "<p>cached prose</p>"
    # Version-keyed: an AN1 version bump must regenerate rather than reuse stale prose.
    assert rebuilt.get_cached_prose("123", "2.1") is None


def test_stale_prose_cache_entries_are_dropped(tmp_path: Path):
    (tmp_path / "an1_prose_cache.json").write_text(
        json.dumps(
            [
                {"post_id": "old", "version": "1.0", "prose_html": "<p>stale</p>",
                 "created_at": "2020-01-01T00:00:00+00:00"},
                {"post_id": "new", "version": "1.0", "prose_html": "<p>fresh</p>",
                 "created_at": datetime.now(UTC).isoformat()},
            ]
        ),
        encoding="utf-8",
    )
    db = HistoryDB(tmp_path / "history.db", prose_cache_ttl_days=14)

    assert db.get_cached_prose("old", "1.0") is None
    assert db.get_cached_prose("new", "1.0") == "<p>fresh</p>"


def test_corrupt_prose_cache_is_recoverable_not_fatal(tmp_path: Path):
    """A corrupt ledger fails the run; a corrupt cache only costs a regeneration."""
    (tmp_path / "an1_prose_cache.json").write_text("{not json", encoding="utf-8")

    db = HistoryDB(tmp_path / "history.db")
    assert db.get_cached_prose("123", "1.0") is None
