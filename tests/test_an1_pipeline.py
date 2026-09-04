from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.db.history import HistoryDB
from src.generators.an1_formatter import AN1Formatter
from src.main import app
from src.scrapers.an1_scraper import (
    AN1Post,
    AN1Scraper,
    AN1ValidationError,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
    post_html = (FIXTURES_DIR / "sample_post.html").read_text(encoding="utf-8")
    dw_html = (FIXTURES_DIR / "sample_dw.html").read_text(encoding="utf-8")

    scraper = AN1Scraper(request_delay=0, verify_direct_link=True)

    def mock_get(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        if "file_" in url:
            mock_resp.text = dw_html
        else:
            mock_resp.text = post_html
        return mock_resp

    def mock_head(url, *args, **kwargs):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return mock_resp

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

    # Primary button must link to the stable AN1 download page
    assert f'href="{sample_post.dw_page_url}" class="agha-btn agha-btn-primary"' in html_content

    # Secondary mirror must link to the direct APK
    assert f'href="{sample_post.direct_download_url}" class="agha-btn agha-btn-secondary"' in html_content

    # Unverified claims must be removed
    assert "Verified Clean • No Malware" not in html_content
    assert "verified to be virus-free" not in html_content


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

    with patch("src.main.AN1Scraper.scrape_post", return_value=sample_post), \
         patch("src.main.AN1Scraper.validate_post", return_value=None), \
         patch("src.main.DB_PATH", tmp_path / "history.db"):
        result = runner.invoke(
            app,
            ["an1-post", sample_post.url, "--dry-run"],
        )

    assert result.exit_code == 0
    assert "Scraped & Formatted" in result.output
    assert "Subway Surfers" in result.output
    assert "Primary Link" in result.output
