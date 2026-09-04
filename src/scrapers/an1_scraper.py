from __future__ import annotations

import logging
import re
import time
import urllib.parse
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger(__name__)


class AN1ScraperError(RuntimeError):
    """Base exception for AN1 scraping errors."""
    pass


class AN1PostNotFoundError(AN1ScraperError):
    """Raised when an AN1 post could not be retrieved or parsed."""
    pass


class AN1ValidationError(AN1ScraperError):
    """Raised when a scraped post fails validation checks."""
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
    dw_page_url: str | None = None
    direct_download_url: str | None = None


class AN1Scraper:
    """Scrapes post listings, article details, and resolved download links from AN1.com."""

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        base_url: str = "https://an1.com",
        session: requests.Session | None = None,
        timeout: int = 15,
        request_delay: float = 1.0,
        verify_direct_link: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.request_delay = request_delay
        self.verify_direct_link = verify_direct_link
        self.session = session or self._create_resilient_session()

    def _create_resilient_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": self.DEFAULT_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @staticmethod
    def extract_post_id(url: str) -> str:
        """Extract post identifier from url (e.g., 'https://an1.com/4683-subway-surfers...' -> '4683')."""
        match = re.search(r"/(\d+)-", url)
        if match:
            return match.group(1)
        slug = url.split("/")[-1].replace(".html", "")
        return slug

    def _extract_next_page_url(self, soup: BeautifulSoup, current_url: str) -> str | None:
        """Extract next page URL from navigation or pagination elements."""
        nav_more = soup.select_one("div.nav_more a")
        if nav_more and nav_more.get("href"):
            return urllib.parse.urljoin(current_url, nav_more["href"])

        next_arrow = soup.select_one("div.navigation_ext a:has(svg.i__arrowright)")
        if next_arrow and next_arrow.get("href"):
            return urllib.parse.urljoin(current_url, next_arrow["href"])

        for a in soup.select("div.navigation a, div.pagination a"):
            text = a.get_text(strip=True).lower()
            if "next" in text or "more" in text:
                href = a.get("href")
                if href:
                    return urllib.parse.urljoin(current_url, href)

        return None

    def fetch_latest_post_urls(
        self,
        limit: int = 20,
        sources: list[str] | None = None,
        max_pages_per_source: int = 10,
    ) -> list[str]:
        """Fetch unique latest post URLs from key AN1 pages with pagination support.

        Defaults to crawling the modded APKs tag page (https://an1.com/tags/mods/).
        """
        if sources is None:
            sources = [
                f"{self.base_url}/tags/mods/",
            ]

        discovered_urls: list[str] = []
        seen: set[str] = set()

        for source_url in sources:
            current_url: str | None = source_url
            page_count = 0

            while current_url and len(discovered_urls) < limit and page_count < max_pages_per_source:
                if self.request_delay > 0 and (seen or page_count > 0):
                    time.sleep(self.request_delay)

                page_count += 1
                try:
                    resp = self.session.get(current_url, timeout=self.timeout)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    logger.warning("Failed to fetch discovery page %s: %s", current_url, exc)
                    break

                soup = BeautifulSoup(resp.text, "lxml")
                links = soup.select("div.data div.name a, div.item_app div.name a, div.app_list div.name a")
                new_links_on_page = 0

                for link in links:
                    href = link.get("href")
                    if not href:
                        continue
                    full_url = urllib.parse.urljoin(self.base_url, href)
                    if re.search(r"/\d+-[^/]+\.html$", full_url) and full_url not in seen:
                        seen.add(full_url)
                        discovered_urls.append(full_url)
                        new_links_on_page += 1
                        if len(discovered_urls) >= limit:
                            return discovered_urls

                if new_links_on_page == 0:
                    break

                next_url = self._extract_next_page_url(soup, current_url)
                if not next_url or next_url == current_url:
                    break
                current_url = next_url

        return discovered_urls

    def scrape_post(self, url: str, *, resolve_download: bool = True) -> AN1Post:
        """Scrape post details and the download page redirect.

        Pass resolve_download=False to skip fetching the /file_*-dw.html page and the
        direct-link HEAD check. Callers that de-duplicate against the publication ledger
        should do that, then call resolve_download_link() only for posts they will publish,
        so already-seen posts cost one request instead of three.
        """
        if self.request_delay > 0:
            time.sleep(self.request_delay)

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
        version = ""
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
            if "Version:" in text and not version:
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
                    full_src = src.replace("/thumbs/", "/")
                    screenshots.append(urllib.parse.urljoin(self.base_url, full_src))

        # Download Page URL
        dw_btn = soup.find("a", class_="download_line") or soup.find("a", class_="btn-green")
        dw_page_url: str | None = None
        if dw_btn and dw_btn.get("href"):
            dw_page_url = urllib.parse.urljoin(self.base_url, dw_btn["href"])

        # Fetch direct download link from download page
        direct_download_url: str | None = None
        if resolve_download and dw_page_url:
            direct_download_url = self._extract_direct_download_link(dw_page_url)

        return AN1Post(
            post_id=post_id,
            url=url,
            title=raw_title,
            app_name=app_name,
            icon_url=icon_url,
            developer=developer,
            categories=categories,
            # Left empty when parsing fails so validate_post() rejects the post instead of
            # silently stamping every post with the same placeholder version.
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

    def _extract_direct_download_link(self, dw_page_url: str) -> str | None:
        """Visit the download page (e.g. /file_*-dw.html) and locate the real direct download link."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

        try:
            resp = self.session.get(dw_page_url, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch download page %s: %s", dw_page_url, exc)
            return None

        soup = BeautifulSoup(resp.text, "lxml")
        candidate_url: str | None = None

        # 1. Check <a id="pre_download" href="...">
        pre_dw = soup.find("a", id="pre_download")
        if pre_dw and pre_dw.get("href") and pre_dw["href"] not in ("#", ""):
            candidate_url = urllib.parse.urljoin(dw_page_url, pre_dw["href"])

        # 2. Check for any direct apk link
        if not candidate_url:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.endswith(".apk") and "an1store.apk" not in href:
                    candidate_url = urllib.parse.urljoin(dw_page_url, href)
                    break

        # 3. Check regex in page body for files.an1 link
        if not candidate_url:
            match = re.search(r'https?://files\.an1\.(?:net|co)/[^\'"\s]+\.apk', resp.text)
            if match:
                candidate_url = match.group(0)

        if not candidate_url:
            return None

        # Optional HEAD check to verify direct link is active and not returning 403/404
        if self.verify_direct_link:
            try:
                head_resp = self.session.head(candidate_url, timeout=5, allow_redirects=True)
                if head_resp.status_code not in (200, 206, 301, 302):
                    logger.warning(
                        "Direct download link %s returned status %d; ignoring.",
                        candidate_url,
                        head_resp.status_code,
                    )
                    return None
            except requests.RequestException as exc:
                logger.warning("Direct link HEAD check failed for %s: %s; ignoring.", candidate_url, exc)
                return None

        return candidate_url

    def resolve_download_link(self, post: AN1Post) -> AN1Post:
        """Attach the direct APK link to a post scraped with resolve_download=False."""
        if post.direct_download_url or not post.dw_page_url:
            return post
        post.direct_download_url = self._extract_direct_download_link(post.dw_page_url)
        return post

    def validate_post(self, post: AN1Post) -> None:
        """Strict validation gate: ensure post is completely and reliably parsed before publishing."""
        errors: list[str] = []

        if not post.post_id:
            errors.append("Missing post_id")
        if not post.title:
            errors.append("Missing post title")
        if not post.app_name or post.app_name == "Android App":
            errors.append(f"Invalid or sentinel app_name: {post.app_name!r}")
        if not post.version or post.version in ("Latest", "Unknown"):
            errors.append(f"Invalid or sentinel version: {post.version!r}")
        if not post.dw_page_url or not post.dw_page_url.startswith(("http://", "https://")):
            errors.append(f"Invalid or missing dw_page_url: {post.dw_page_url!r}")

        if errors:
            raise AN1ValidationError(f"Validation failed for post '{post.title}': {'; '.join(errors)}")
