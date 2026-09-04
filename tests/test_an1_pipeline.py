from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.db.history import HistoryDB
from src.generators.an1_formatter import AN1Formatter
from src.main import app
from src.scrapers.an1_scraper import AN1Post, AN1Scraper


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
    scraper = AN1Scraper()
    assert scraper.extract_post_id("https://an1.com/4683-subway-surfers-mod-apk-8.html") == "4683"
    assert scraper.extract_post_id("https://an1.com/266-sniper-3d.html") == "266"
    assert scraper.extract_post_id("https://an1.com/custom-game.html") == "custom-game"


def test_formatter_generates_valid_html_and_links(sample_post: AN1Post):
    formatter = AN1Formatter()
    html = formatter.format_html(sample_post)
    title = formatter.build_post_title(sample_post)
    labels = formatter.build_labels(sample_post)

    assert "Subway Surfers" in title
    assert "MOD APK" in labels
    assert "Arcade" in labels

    # Check that both direct download and dw_page URLs are embedded
    assert sample_post.direct_download_url in html
    assert sample_post.dw_page_url in html
    assert sample_post.icon_url in html
    assert "252.1Mb" in html
    assert "Android 6.0+" in html
    assert "SYBO Games" in html


def test_history_db_an1_tracking(tmp_path: Path, sample_post: AN1Post):
    db_path = tmp_path / "test_history.db"
    json_path = tmp_path / "an1_published.json"
    history = HistoryDB(db_path, json_tracker_path=json_path)

    assert not history.is_an1_published(sample_post.post_id)
    assert len(history.get_published_an1_ids()) == 0

    history.record_an1_publication(
        post_id=sample_post.post_id,
        source_url=sample_post.url,
        title=sample_post.title,
        direct_download_url=sample_post.direct_download_url,
        dw_page_url=sample_post.dw_page_url,
        blogger_post_id="post-12345",
        blogger_url="https://example.blogspot.com/post.html",
        status="LIVE",
    )

    assert history.is_an1_published(sample_post.post_id)
    assert sample_post.post_id in history.get_published_an1_ids()
    assert json_path.exists()

    # Verify JSON file has been written correctly
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["post_id"] == "4683"
    assert data[0]["direct_download_url"] == sample_post.direct_download_url

    # Test seed/sync in a fresh DB instance
    fresh_db_path = tmp_path / "fresh_history.db"
    fresh_history = HistoryDB(fresh_db_path, json_tracker_path=json_path)
    assert fresh_history.is_an1_published("4683")


def test_cli_an1_post_dry_run(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["an1-post", "https://an1.com/4683-subway-surfers-mod-apk-8.html", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "Scraped & Formatted" in result.output
    assert "files.an1.net" in result.output
