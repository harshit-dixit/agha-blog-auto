from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup


class AN1ScraperError(RuntimeError):
    """Base exception for AN1 scraping errors."""
    pass


class AN1PostNotFoundError(AN1ScraperError):
    """Raised when an AN1 post could not be retrieved or parsed."""
    pass


@dataclass
class AN1Post:
    post_id: str
    url: str
    title: str
    app_name: str
    icon_url: str
    developer: str
    categories: list[str]
    version: str
    android_version: str
    size: str
    updated_date: str
    rating: str
    installs: str
    mod_features: str
    description_html: str
    description_text: str
    screenshots: list[str] = field(default_factory=list)
    dw_page_url: Optional[str] = None
    direct_download_url: Optional[str] = None


class AN1Scraper:
    """Scrapes post listings, article details, and resolved download links from AN1.com."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        base_url: str = "https://an1.com",
        session: Optional[requests.Session] = None,
        timeout: int = 15,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    @staticmethod
    def extract_post_id(url: str) -> str:
        """Extract post identifier from url (e.g., 'https://an1.com/4683-subway-surfers...' -> '4683')."""
        match = re.search(r"/(\d+)-", url)
        if match:
            return match.group(1)
        # Fallback to slug without extension
        slug = url.split("/")[-1].replace(".html", "")
        return slug

    def fetch_latest_post_urls(
        self,
        limit: int = 20,
        sources: Optional[list[str]] = None,
    ) -> list[str]:
        """Fetch unique latest post URLs from key AN1 pages."""
        if sources is None:
            sources = [
                f"{self.base_url}/",
                f"{self.base_url}/games/",
                f"{self.base_url}/programmy/",
            ]

        discovered_urls: list[str] = []
        seen: set[str] = set()

        for source_url in sources:
            try:
                resp = self.session.get(source_url, timeout=self.timeout)
                resp.raise_for_status()
            except requests.RequestException as exc:
                continue

            soup = BeautifulSoup(resp.text, "lxml")
            # Anchor selectors found in post cards across an1.com
            links = soup.select("div.data div.name a, div.item_app div.name a, div.app_list div.name a")
            for link in links:
                href = link.get("href")
                if not href:
                    continue
                full_url = urllib.parse.urljoin(self.base_url, href)
                # Ensure it's a content post (e.g. /<digits>-slug.html)
                if re.search(r"/\d+-[^/]+\.html$", full_url):
                    if full_url not in seen:
                        seen.add(full_url)
                        discovered_urls.append(full_url)
                        if len(discovered_urls) >= limit:
                            return discovered_urls

        return discovered_urls

    def scrape_post(self, url: str) -> AN1Post:
        """Scrape full post details, download page redirect, and direct APK link."""
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise AN1PostNotFoundError(f"Failed to fetch post from {url}: {exc}") from exc

        soup = BeautifulSoup(resp.text, "lxml")
        post_id = self.extract_post_id(url)

        # Title
        title_el = soup.find("h1", class_="title")
        raw_title = title_el.get_text(strip=True) if title_el else ""

        # App Name (from schema or parsed title)
        name_meta = soup.find("meta", itemprop="name")
        if name_meta and name_meta.get("content"):
            app_name = name_meta["content"].strip()
        else:
            # Fallback parse: "Download <AppName> (MOD, ...) 1.2.3 free on android"
            clean = re.sub(r"^Download\s+", "", raw_title, flags=re.I)
            clean = re.sub(r"\s+\(MOD.*$", "", clean, flags=re.I)
            clean = re.sub(r"\s+\d+(\.\d+)+.*$", "", clean)
            app_name = clean.strip() or "Android App"

        # MOD Features extraction from title
        mod_match = re.search(r"\((MOD[^)]*)\)", raw_title, re.I)
        mod_features = mod_match.group(1).strip() if mod_match else "MOD APK"

        # App Icon
        icon_figure = soup.find("figure", class_="img")
        icon_img = icon_figure.find("img") if icon_figure else soup.find("img", itemprop="image")
        icon_url = ""
        if icon_img and icon_img.get("src"):
            icon_url = urllib.parse.urljoin(self.base_url, icon_img["src"])

        # Developer
        dev_el = soup.find("div", class_="developer")
        developer = dev_el.get_text(strip=True) if dev_el else "Unknown Developer"

        # Categories
        cat_links = soup.select("ul.catbar li a")
        categories = [
            a.get_text(strip=True)
            for a in cat_links
            if a.get_text(strip=True).lower() not in ("an1.com", "home")
        ]
        if not categories:
            categories = ["Games", "MOD"]

        # Specs from spec lists
        version = "Latest"
        android_version = "Android 6.0+"
        size = "Unknown"
        updated_date = "Recently"
        installs = "1,000,000+"
        rating = "4.5"

        ver_meta = soup.find("span", itemprop="softwareVersion")
        if ver_meta:
            version = ver_meta.get_text(strip=True)

        size_meta = soup.find("span", itemprop="fileSize")
        if size_meta:
            size = size_meta.get_text(strip=True)

        os_meta = soup.find("span", itemprop="operatingSystem")
        if os_meta:
            android_version = os_meta.get_text(strip=True)

        for li in soup.select("ul.spec li"):
            text = li.get_text(" ", strip=True)
            if "Version:" in text and version == "Latest":
                version = text.replace("Version:", "").strip()
            elif ("Mb" in text or "Gb" in text or "MB" in text) and size == "Unknown":
                size = text.strip()
            elif "Android" in text and android_version == "Android 6.0+":
                android_version = text.strip()
            elif "Updated" in text:
                updated_date = text.replace("Updated", "").strip()
            elif "Installs" in text:
                installs = text.replace("Installs", "").strip()

        rate_val = soup.find("span", itemprop="ratingValue")
        if rate_val:
            rating = rate_val.get_text(strip=True)

        # Description
        desc_div = soup.find("div", id="spoiler") or soup.find("div", class_="description")
        description_text = ""
        description_html = ""
        if desc_div:
            # Extract clean text and inner HTML
            description_text = desc_div.get_text(" ", strip=True)
            description_html = "".join(
                str(child) for child in desc_div.children if getattr(child, "name", None) != "button"
            ).strip()

        # Screenshots
        screenshots: list[str] = []
        screen_links = soup.select("div.app_screens_list a")
        for a in screen_links:
            href = a.get("href")
            if href:
                screenshots.append(urllib.parse.urljoin(self.base_url, href))
        if not screenshots:
            for img in soup.select("div.app_screens_list img"):
                src = img.get("src")
                if src:
                    # Convert thumbnail to full image if thumbnail URL pattern exists
                    full_src = src.replace("/thumbs/", "/")
                    screenshots.append(urllib.parse.urljoin(self.base_url, full_src))

        # Download Page URL
        dw_btn = soup.find("a", class_="download_line") or soup.find("a", class_="btn-green")
        dw_page_url: Optional[str] = None
        if dw_btn and dw_btn.get("href"):
            dw_page_url = urllib.parse.urljoin(self.base_url, dw_btn["href"])

        # Fetch direct download link from download page
        direct_download_url: Optional[str] = None
        if dw_page_url:
            direct_download_url = self._extract_direct_download_link(dw_page_url)

        return AN1Post(
            post_id=post_id,
            url=url,
            title=raw_title,
            app_name=app_name,
            icon_url=icon_url,
            developer=developer,
            categories=categories,
            version=version,
            android_version=android_version,
            size=size,
            updated_date=updated_date,
            rating=rating,
            installs=installs,
            mod_features=mod_features,
            description_html=description_html,
            description_text=description_text,
            screenshots=screenshots,
            dw_page_url=dw_page_url,
            direct_download_url=direct_download_url,
        )

    def _extract_direct_download_link(self, dw_page_url: str) -> Optional[str]:
        """Visit the download page (e.g. /file_*-dw.html) and locate the real direct download link."""
        try:
            resp = self.session.get(dw_page_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # 1. Check <a id="pre_download" href="...">
        pre_dw = soup.find("a", id="pre_download")
        if pre_dw and pre_dw.get("href") and pre_dw["href"] not in ("#", ""):
            return pre_dw["href"]

        # 2. Check for any direct apk link
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".apk") and "an1store.apk" not in href:
                return href

        # 3. Check regex in page body for files.an1 link
        match = re.search(r'https?://files\.an1\.(?:net|co)/[^\'"\s]+\.apk', resp.text)
        if match:
            return match.group(0)

        return None
