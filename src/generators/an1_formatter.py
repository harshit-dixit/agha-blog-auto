from __future__ import annotations

import html
import json
import logging
import re
from collections.abc import Sequence

from google import genai
from google.genai import types

from config.settings import Settings
from src.scrapers.an1_scraper import AN1Post

logger = logging.getLogger(__name__)

# Minimum usable length for generated prose; shorter output means the model bailed out.
MIN_ENHANCED_LENGTH = 200

# Ceiling for a batched reply. Four-paragraph reviews run 550-800 tokens each once the
# HTML is JSON-escaped, so this is the headroom that keeps a small batch from truncating.
BATCH_MAX_OUTPUT_TOKENS = 8192


class ContentEnhancementError(RuntimeError):
    """Raised when Gemini is configured but could not produce usable original prose.

    Publishing the scraped description verbatim is the failure mode this guards against,
    so callers should skip the post rather than fall back to it.
    """


class QuotaExhaustedError(ContentEnhancementError):
    """Raised when Gemini refuses the request because the API quota is spent.

    This is a run-level condition, not a per-post one: every later call in the same run
    fails the same way, so callers should stop generating instead of walking the rest of
    the backlog burning requests. Subclasses ContentEnhancementError so existing per-post
    handlers still treat it as a skip.
    """


GEMINI_ENHANCE_PROMPT = """You are a senior gaming journalist and app reviewer for Android Game Hack Area.
Write a comprehensive, engaging, and original app review for the following Android game/application based on its metadata:

App Name: {app_name}
Version: {version}
Developer: {developer}
Category: {category}
Mod Features: {mod_features}
Scraped Summary: {raw_description}

INSTRUCTIONS:
1. Write 3-4 detailed, beautifully phrased paragraphs covering:
   - Introduction: What the app/game is, its premise, why it's popular, and what makes it fun.
   - Core Mechanics & Gameplay: Controls, graphics, atmosphere, game modes, progression.
   - MOD Highlights: Explain what the modified features ({mod_features}) unlock and how they enhance the gameplay.
   - Player Tips: 2-3 helpful strategies or settings recommendations.
2. Tone: Engaging, technical yet accessible, gamer-friendly.
3. Output format: Return ONLY the paragraphs wrapped in clean HTML <p>...</p> tags.
4. Do NOT include markdown fences, <html>, <body>, <h1>, or download links (these are handled by the template).
5. Ensure the text is 100% original and natural (not a regurgitation of the scraped summary)."""

# Reviews for several posts in one request. The free tier's binding limit is a per-day
# request count (200/day on Flash-Lite, 20/day on Flash), not a per-minute rate, so how
# many calls a run makes matters more than how fast it makes them.
GEMINI_BATCH_PROMPT = """You are a senior gaming journalist and app reviewer for Android Game Hack Area.
Write an original app review for EACH of the {count} Android games listed below.

{app_blocks}

INSTRUCTIONS (apply to every review independently):
1. Write 3-4 detailed paragraphs covering:
   - Introduction: what the game is, its premise, why it's popular, and what makes it fun.
   - Core Mechanics & Gameplay: controls, graphics, atmosphere, game modes, progression.
   - MOD Highlights: what that entry's listed mod features unlock and how they change play.
   - Player Tips: 2-3 helpful strategies or settings recommendations.
2. Tone: engaging, technical yet accessible, gamer-friendly.
3. Write each review fresh for its own game. Do NOT reuse opening phrases, sentence
   structures, or stock transitions between entries - these posts are published side by
   side, so repeated phrasing is obvious to both readers and search engines.
4. Every review must be 100% original, not a rewording of that game's scraped summary.
5. Return ONLY a JSON array - no markdown fences, no commentary - shaped exactly like:
   [{{"post_id": "<the id given for that game>", "review_html": "<p>...</p><p>...</p>"}}]
6. review_html must contain only <p>...</p> paragraphs: no markdown, no <html>, <body> or
   <h1> tags, and no download links (the template adds those).
7. Return exactly one entry per game, reusing the exact post_id given for that game."""


class AN1Formatter:
    """Formats an AN1 scraped post into a modern, mobile-responsive HTML post for Blogger."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self._genai_client = None
        if self.settings and self.settings.GEMINI_API_KEY:
            try:
                self._genai_client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
            except Exception as exc:
                logger.warning("Failed to initialize Gemini client for AN1Formatter: %s", exc)

    def build_post_title(self, post: AN1Post) -> str:
        """Create a clean, SEO-optimized title for Blogger."""
        features_snippet = f" ({post.mod_features})" if post.mod_features else ""
        ver_snippet = f" v{post.version}" if post.version and post.version.lower() != "latest" else ""
        title = f"{post.app_name}{ver_snippet}{features_snippet} Download for Android"
        if len(title) > 90:
            title = f"{post.app_name}{ver_snippet} MOD APK Download"
        return title

    def build_labels(self, post: AN1Post) -> list[str]:
        """Generate relevant Blogger tags/labels."""
        labels: list[str] = ["MOD APK", "Android Games"]
        if post.app_name:
            labels.append(post.app_name)
        for cat in post.categories:
            if cat and cat not in labels and len(cat) < 25:
                labels.append(cat)
        if "MOD" in post.mod_features.upper() or "UNLIMITED" in post.mod_features.upper():
            labels.append("Unlimited Money")
        labels.append("Direct Download")
        return list(dict.fromkeys(labels))[:8]

    @property
    def gemini_enabled(self) -> bool:
        """True when a Gemini client was built, so original prose can be generated."""
        return self._genai_client is not None

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        """Detect quota exhaustion (HTTP 429 / RESOURCE_EXHAUSTED) in an SDK exception.

        Checked structurally first, then against the message, because the SDK surfaces
        the same condition through several exception shapes depending on transport.
        """
        if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
            return True
        status = getattr(exc, "status", "")
        if isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED":
            return True
        text = str(exc).upper()
        return "RESOURCE_EXHAUSTED" in text or "429" in text

    @classmethod
    def _generation_error(cls, subject: str, exc: Exception) -> ContentEnhancementError:
        """Wrap an SDK failure, tagging quota exhaustion so callers can stop the run."""
        message = f"Gemini enhancement failed for {subject}: {exc}"
        if cls._is_quota_error(exc):
            return QuotaExhaustedError(message)
        return ContentEnhancementError(message)

    def _enhance_with_gemini(self, post: AN1Post) -> str | None:
        """Generate original review prose via Gemini.

        Returns None only when Gemini is not configured. When it *is* configured, a failed
        or unusable generation raises ContentEnhancementError instead of quietly falling
        back to the scraped description, which would republish AN1's text verbatim.
        """
        if not self._genai_client or not self.settings:
            return None

        prompt = GEMINI_ENHANCE_PROMPT.format(
            app_name=post.app_name,
            version=post.version,
            developer=post.developer,
            category=", ".join(post.categories),
            mod_features=post.mod_features,
            raw_description=post.description_text,
        )

        try:
            response = self._genai_client.models.generate_content(
                model=self.settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            raw_text = response.text or ""
        except Exception as exc:
            raise self._generation_error(repr(post.app_name), exc) from exc

        text = self._normalize_enhanced_text(raw_text)
        if not text:
            raise ContentEnhancementError(
                f"Gemini returned unusable prose for {post.app_name!r} "
                f"({len(raw_text.strip())} chars, no usable <p> block)"
            )
        return text

    @staticmethod
    def _normalize_enhanced_text(raw_text: str) -> str | None:
        """Strip fences and any preamble, returning usable paragraph HTML or None."""
        text = raw_text.replace("```html", "").replace("```", "").strip()

        # Drop any lead-in prose before the first paragraph tag ("Here is the review:" etc.)
        first_p = re.search(r"<p[\s>]", text, re.I)
        if first_p:
            text = text[first_p.start():].strip()

        if not text.lower().startswith("<p") or len(text) < MIN_ENHANCED_LENGTH:
            return None
        return text

    def enhance_batch(self, posts: Sequence[AN1Post]) -> dict[str, str]:
        """Generate review prose for several posts in a single Gemini request.

        Returns a {post_id: paragraph_html} map covering only the entries that came back
        usable. Callers must skip any post missing from the map rather than publishing its
        scraped description - a skipped post stays in the backlog and is retried next run,
        which costs a delay instead of a thin duplicate-content page. Returns an empty map
        when Gemini is not configured.

        Raises QuotaExhaustedError when the API is out of quota (the caller should stop
        generating) and ContentEnhancementError for any other failure (skip this batch).
        """
        if not self.gemini_enabled or not self.settings or not posts:
            return {}

        blocks = []
        for index, post in enumerate(posts, start=1):
            blocks.append(
                "\n".join(
                    (
                        f"GAME {index}",
                        f"post_id: {post.post_id}",
                        f"App Name: {post.app_name}",
                        f"Version: {post.version}",
                        f"Developer: {post.developer}",
                        f"Category: {', '.join(post.categories)}",
                        f"Mod Features: {post.mod_features}",
                        f"Scraped Summary: {post.description_text}",
                    )
                )
            )
        prompt = GEMINI_BATCH_PROMPT.format(count=len(posts), app_blocks="\n\n".join(blocks))

        try:
            response = self._genai_client.models.generate_content(
                model=self.settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    response_mime_type="application/json",
                    max_output_tokens=BATCH_MAX_OUTPUT_TOKENS,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
            raw_text = response.text or ""
        except Exception as exc:
            subject = f"batch of {len(posts)} ({', '.join(p.app_name for p in posts)})"
            raise self._generation_error(subject, exc) from exc

        wanted = {post.post_id: post for post in posts}
        results: dict[str, str] = {}
        for item in self._parse_batch_items(raw_text):
            post_id = str(item.get("post_id", "")).strip()
            if post_id not in wanted or post_id in results:
                continue
            text = self._normalize_enhanced_text(str(item.get("review_html") or ""))
            if text:
                results[post_id] = text

        missing = [post.app_name for pid, post in wanted.items() if pid not in results]
        if missing:
            logger.warning(
                "Gemini batch returned no usable prose for %d of %d post(s): %s",
                len(missing),
                len(posts),
                ", ".join(missing),
            )
        if not results:
            raise ContentEnhancementError(
                f"Gemini batch returned no usable reviews for any of the {len(posts)} "
                f"post(s) ({len(raw_text.strip())} chars returned)"
            )
        return results

    @staticmethod
    def _parse_batch_items(raw_text: str) -> list[dict]:
        """Parse a batch reply into entries, salvaging whole ones from a truncated array.

        A cut-off or malformed reply should cost only the entries it actually mangled, so
        the fallback pass recovers every complete JSON object it can find instead of
        discarding the whole batch.
        """
        text = raw_text.replace("```json", "").replace("```", "").strip()
        start = text.find("[")
        if start > 0:
            text = text[start:]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            # Some replies wrap the array in a single key, e.g. {"reviews": [...]}.
            for value in parsed.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [parsed]

        items: list[dict] = []
        for match in re.finditer(r"\{[^{}]*\}", text, re.S):
            try:
                candidate = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                items.append(candidate)
        return items

    def format_html(self, post: AN1Post, enhanced_desc: str | None = None) -> str:
        """Generate modern, fully responsive, standalone-styled HTML for Blogger post body."""
        app_name = html.escape(post.app_name)
        developer = html.escape(post.developer)
        version = html.escape(post.version)
        android_ver = html.escape(post.android_version)
        size = html.escape(post.size)
        updated = html.escape(post.updated_date)
        mod_features = html.escape(post.mod_features)
        rating = html.escape(post.rating or "4.8")
        installs = html.escape(post.installs or "1,000,000+")

        # File size is dropped from button and CTA labels when parsing produced the sentinel,
        # so a button never reads "Download APK (Unknown)".
        size_known = bool(post.size) and post.size.strip().lower() not in ("unknown", "n/a", "-")
        size_label = f" ({size})" if size_known else ""
        size_meta_line = f"Version {version} • File Size {size}" if size_known else f"Version {version}"

        # Primary download link: Stable AN1 download page
        primary_link = post.dw_page_url or post.url
        # Secondary download link: Direct APK mirror (if resolved and verified)
        mirror_link = post.direct_download_url if (post.direct_download_url and post.direct_download_url != primary_link) else None

        # Build download buttons (Primary: near-black #0a0a0a; Secondary mirror: pink #ff4d8b)
        primary_btn_html = f"""
        <a href="{html.escape(primary_link)}" class="dl-btn" target="_blank" rel="noopener nofollow">
            <span>Download APK{size_label}</span>
        </a>
        """

        mirror_btn_html = ""
        if mirror_link:
            mirror_btn_html = f"""
            <a href="{html.escape(mirror_link)}" class="dl-btn alt" target="_blank" rel="noopener nofollow">
                <span>⚡ Direct APK Mirror</span>
            </a>
            """

        if mirror_link:
            btn_group_html = f"""
            <div class="dl-row">
                {primary_btn_html}
                {mirror_btn_html}
            </div>
            """
        else:
            btn_group_html = f"""
            <div class="dl-wrap">
                {primary_btn_html}
            </div>
            """

        # Build screenshots HTML with theme-native .screens snap-scroller
        screenshots_html = ""
        if post.screenshots:
            items = []
            for img_url in post.screenshots[:6]:
                safe_img = html.escape(img_url)
                items.append(
                    f'<img src="{safe_img}" alt="{app_name} Screenshot" loading="lazy" />'
                )
            screenshots_html = f"""
            <div class="post-section">
                <h2>📸 In-Game Screenshots</h2>
                <div class="screens">
                    {''.join(items)}
                </div>
            </div>
            """

        # Original prose either arrives pre-generated from a batch call upstream, or is
        # generated per-post here. The scraped-text fallback below is only reached when no
        # Gemini key is set (a failed generation raises instead).
        if enhanced_desc is None:
            enhanced_desc = self._enhance_with_gemini(post)
        if enhanced_desc:
            desc_formatted = enhanced_desc
        else:
            desc_text = post.description_text or "Download the latest modified version with unlocked features and unlimited resources."
            raw_blocks = [b.strip() for b in desc_text.split("\n\n") if b.strip()]
            if not raw_blocks:
                raw_blocks = [desc_text]
            desc_formatted = "".join(f"<p>{html.escape(b)}</p>" for b in raw_blocks)

        html_body = f"""
<div class="post-content agha-clay-post">
    <style>
        :root, .agha-clay-post {{
            --canvas: #fffaf0;
            --soft: #faf5e8;
            --card: #f5f0e0;
            --strong: #ebe6d6;
            --hair: #e5e5e5;
            --ink: #0a0a0a;
            --body-strong: #1a1a1a;
            --body: #3a3a3a;
            --muted: #6a6a6a;
            --muted-soft: #9a9a9a;
            --on-dark: #ffffff;
            --primary: #0a0a0a;
            --pink: #ff4d8b;
            --teal: #1a3a3a;
            --lav: #b8a4ed;
            --peach: #ffb084;
            --ochre: #e8b94a;
            --mint: #a4d4c5;
            --coral: #ff6b5a;
            --ok: #22c55e;
            --warn: #f59e0b;
            --err: #ef4444;
            --r-xs: 6px;
            --r-sm: 8px;
            --r-md: 12px;
            --r-lg: 16px;
            --r-xl: 24px;
            --r-pill: 9999px;
            --r-full: 50%;
            --sans: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            --lift: 0 8px 24px rgba(10, 26, 26, 0.08);
        }}
        .agha-clay-post {{
            font-family: var(--sans);
            color: var(--body);
            font-size: 17px;
            line-height: 1.7;
            max-width: 800px;
            margin: 0 auto;
            -webkit-font-smoothing: antialiased;
        }}
        .agha-clay-post h2 {{
            font-size: 24px;
            font-weight: 500;
            line-height: 1.25;
            letter-spacing: -0.025em;
            color: var(--ink);
            margin: 1.8em 0 0.6em;
        }}
        .agha-clay-post p {{
            margin: 0 0 1.15em;
            color: var(--body);
        }}

        /* App Header Card (.app-head) */
        .agha-clay-post .app-head {{
            display: flex;
            align-items: center;
            gap: 16px;
            background: var(--card);
            border: 1px solid var(--hair);
            border-radius: var(--r-lg);
            padding: 18px;
            margin: 0 0 24px 0;
        }}
        .agha-clay-post .app-icon {{
            width: 76px;
            height: 76px;
            border-radius: var(--r-md);
            object-fit: cover;
            flex: 0 0 auto;
            background: var(--soft);
        }}
        .agha-clay-post .app-head-info {{
            flex: 1;
            min-width: 0;
        }}
        .agha-clay-post .app-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-bottom: 6px;
        }}
        .agha-clay-post .pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: var(--r-pill);
            background: var(--card);
            color: var(--ink);
            font: 500 12px/1.4 var(--sans);
            white-space: nowrap;
        }}
        .agha-clay-post .pill-pink {{
            background: var(--pink);
            color: var(--on-dark);
        }}
        .agha-clay-post .pill-teal {{
            background: var(--teal);
            color: var(--on-dark);
        }}
        .agha-clay-post .pill-ochre {{
            background: var(--ochre);
            color: var(--ink);
        }}
        .agha-clay-post .pill-dot {{
            width: 6px;
            height: 6px;
            border-radius: var(--r-full);
            background: currentColor;
            opacity: 0.7;
        }}
        .agha-clay-post .an {{
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.015em;
            color: var(--ink);
            line-height: 1.25;
            margin: 0 0 4px 0;
        }}
        .agha-clay-post .am {{
            font-size: 13px;
            color: var(--muted);
        }}

        /* Download CTA Card (.dl-card) */
        .agha-clay-post .dl-card {{
            background: var(--card);
            border: 1px solid var(--hair);
            border-radius: var(--r-lg);
            padding: 22px;
            text-align: center;
            margin: 24px 0;
        }}
        .agha-clay-post .dl-card-title {{
            font-size: 18px;
            font-weight: 600;
            color: var(--ink);
            letter-spacing: -0.01em;
            margin-bottom: 4px;
        }}
        .agha-clay-post .dl-card-sub {{
            font-size: 13px;
            color: var(--muted);
        }}
        .agha-clay-post .dl-wrap {{
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-top: 14px;
        }}
        .agha-clay-post .dl-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 14px;
        }}
        .agha-clay-post .dl-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-height: 52px;
            padding: 14px 24px;
            border-radius: var(--r-md);
            background: var(--primary);
            color: var(--on-dark) !important;
            font: 600 15px/1.2 var(--sans);
            text-decoration: none !important;
            text-align: center;
            transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
            cursor: pointer;
        }}
        .agha-clay-post .dl-btn:hover {{
            transform: translateY(-2px);
            box-shadow: var(--lift);
            color: var(--on-dark) !important;
        }}
        .agha-clay-post .dl-btn::after {{
            content: "\\2193";
            font-size: 16px;
            font-weight: 600;
            opacity: 0.9;
        }}
        .agha-clay-post .dl-btn.alt {{
            background: var(--pink);
            color: var(--on-dark) !important;
        }}
        .agha-clay-post .dl-btn.alt:hover {{
            background: #e63e79;
        }}

        /* MOD Callout Notice (.notice) */
        .agha-clay-post .notice {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin: 24px 0;
            padding: 16px 20px;
            background: var(--card);
            border-radius: var(--r-md);
            border-left: 4px solid var(--lav);
            font-size: 15px;
            line-height: 1.6;
        }}
        .agha-clay-post .notice-title {{
            font-weight: 600;
            font-size: 15px;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .agha-clay-post .notice-desc {{
            margin: 0;
            color: var(--body);
            font-size: 14px;
            font-weight: 500;
        }}

        /* App Specifications (.app-box) */
        .agha-clay-post .app-box {{
            background: var(--card);
            border: 1px solid var(--hair);
            border-radius: var(--r-lg);
            padding: 8px 24px;
            margin: 16px 0 24px;
            font-size: 15px;
        }}
        .agha-clay-post .app-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: baseline;
            padding: 12px 0;
            border-bottom: 1px solid rgba(10, 10, 10, 0.08);
        }}
        .agha-clay-post .app-row:last-child {{
            border-bottom: 0;
        }}
        .agha-clay-post .app-row span {{
            flex: 0 0 40%;
            max-width: 190px;
            font-weight: 500;
            color: var(--muted);
        }}
        .agha-clay-post .app-row b {{
            font-weight: 600;
            color: var(--ink);
        }}

        /* Screenshot Scroller (.screens) */
        .agha-clay-post .screens {{
            display: flex;
            gap: 12px;
            overflow-x: auto;
            margin: 16px 0 24px;
            padding-bottom: 8px;
            scroll-snap-type: x mandatory;
            scrollbar-width: none;
        }}
        .agha-clay-post .screens::-webkit-scrollbar {{
            display: none;
        }}
        .agha-clay-post .screens img {{
            flex: 0 0 auto;
            width: 180px;
            height: 320px;
            object-fit: cover;
            scroll-snap-align: start;
            border-radius: var(--r-md);
            border: 1px solid var(--hair);
            margin: 0 !important;
        }}

        /* Installation Steps (.steps) with Ochre Clay Discs */
        .agha-clay-post .steps {{
            counter-reset: st;
            list-style: none;
            margin: 16px 0 24px;
            padding: 0;
        }}
        .agha-clay-post .steps li {{
            position: relative;
            counter-increment: st;
            padding: 0 0 16px 50px;
            font-size: 15px;
            line-height: 1.6;
            color: var(--body);
        }}
        .agha-clay-post .steps li::before {{
            content: counter(st);
            position: absolute;
            left: 0;
            top: -2px;
            width: 34px;
            height: 34px;
            border-radius: var(--r-full);
            background: var(--ochre);
            color: var(--ink);
            display: flex;
            align-items: center;
            justify-content: center;
            font: 600 15px/1 var(--sans);
        }}
        .agha-clay-post .steps li:last-child {{
            padding-bottom: 0;
        }}

        /* FAQ Accordion (.faq) */
        .agha-clay-post .faq {{
            margin: 16px 0 24px;
        }}
        .agha-clay-post .faq details {{
            background: var(--card);
            border: 1px solid var(--hair);
            border-radius: var(--r-md);
            padding: 14px 18px;
            margin-bottom: 10px;
        }}
        .agha-clay-post .faq summary {{
            display: flex;
            align-items: center;
            gap: 10px;
            cursor: pointer;
            list-style: none;
            font: 600 15px/1.45 var(--sans);
            color: var(--ink);
        }}
        .agha-clay-post .faq summary::-webkit-details-marker {{
            display: none;
        }}
        .agha-clay-post .faq summary::after {{
            content: "";
            width: 8px;
            height: 8px;
            border-right: 2px solid var(--muted);
            border-bottom: 2px solid var(--muted);
            transform: rotate(45deg);
            margin-left: auto;
            flex: 0 0 auto;
            transition: transform 0.18s ease;
        }}
        .agha-clay-post .faq details[open] summary::after {{
            transform: rotate(-135deg);
        }}
        .agha-clay-post .faq details > *:not(summary) {{
            margin-top: 10px;
            font-size: 14px;
            line-height: 1.6;
            color: var(--body);
        }}

        /* Pre-Footer Bottom CTA Card */
        .agha-clay-post .bottom-cta {{
            background: var(--soft);
            border: 1px solid var(--hair);
            border-radius: var(--r-xl);
            padding: 30px 24px;
            text-align: center;
            margin: 36px 0 16px;
        }}
        .agha-clay-post .bottom-cta h3 {{
            font-size: 22px;
            font-weight: 500;
            letter-spacing: -0.02em;
            color: var(--ink);
            margin: 0 0 6px 0;
        }}
        .agha-clay-post .bottom-cta p {{
            font-size: 14px;
            color: var(--muted);
            margin: 0 0 16px 0;
        }}
        .agha-clay-post .disclaimer {{
            font-size: 12px !important;
            color: var(--muted-soft) !important;
            margin: 16px 0 0 0 !important;
            line-height: 1.5;
        }}

        @media (max-width: 640px) {{
            .agha-clay-post .app-head {{
                flex-direction: column;
                text-align: center;
            }}
            .agha-clay-post .app-badges {{
                justify-content: center;
            }}
            .agha-clay-post .dl-row {{
                grid-template-columns: 1fr;
            }}
            .agha-clay-post .app-row span {{
                flex: 1 1 100%;
                max-width: none;
            }}
        }}
    </style>

    <!-- App Header Card -->
    <div class="app-head">
        <img class="app-icon" src="{html.escape(post.icon_url)}" alt="{app_name} Icon" loading="lazy" />
        <div class="app-head-info">
            <div class="app-badges">
                <span class="pill pill-pink">MOD APK</span>
                <span class="pill pill-teal">v{version}</span>
                <span class="pill pill-ochre">⭐ {rating}/5</span>
            </div>
            <div class="an">{app_name}</div>
            <div class="am">By <strong>{developer}</strong> • {installs} Downloads</div>
        </div>
    </div>

    <!-- Quick CTA Top -->
    <div class="dl-card">
        <div class="dl-card-title">Get {app_name} MOD APK for Android</div>
        <div class="dl-card-sub">{size_meta_line}</div>
        {btn_group_html}
        <div style="margin-top: 14px;">
            <span class="pill pill-teal"><span class="pill-dot"></span>Fast Direct Download • Verified Safe</span>
        </div>
    </div>

    <!-- MOD Features Notice -->
    <div class="notice">
        <div class="notice-title">✨ Mod Features Unlocked</div>
        <p class="notice-desc">{mod_features}</p>
    </div>

    <!-- Detailed Specifications -->
    <div class="post-section">
        <h2>📋 App Specifications</h2>
        <div class="app-box">
            <div class="app-row">
                <span>App Name</span>
                <b>{app_name}</b>
            </div>
            <div class="app-row">
                <span>Version</span>
                <b>{version}</b>
            </div>
            <div class="app-row">
                <span>File Size</span>
                <b>{size}</b>
            </div>
            <div class="app-row">
                <span>Requires Android</span>
                <b>{android_ver}</b>
            </div>
            <div class="app-row">
                <span>Developer</span>
                <b>{developer}</b>
            </div>
            <div class="app-row">
                <span>Last Updated</span>
                <b>{updated}</b>
            </div>
        </div>
    </div>

    <!-- App Overview & Features -->
    <div class="post-section">
        <h2>📖 About {app_name}</h2>
        {desc_formatted}
    </div>

    <!-- Screenshots -->
    {screenshots_html}

    <!-- Installation Guide -->
    <div class="post-section">
        <h2>📲 How to Install MOD APK</h2>
        <ol class="steps">
            <li><strong>Download APK:</strong> Tap the download button above or below to save the APK file to your Android device.</li>
            <li><strong>Allow Unknown Sources:</strong> If prompted, navigate to <em>Settings &gt; Security (or Apps)</em> and permit installation from Unknown Sources for your browser or file manager.</li>
            <li><strong>Install Package:</strong> Open the downloaded file in your Notification bar or Downloads directory and tap <strong>Install</strong>.</li>
            <li><strong>Launch & Enjoy:</strong> Open {app_name} and explore all unlocked modified features with unlimited resources!</li>
        </ol>
    </div>

    <!-- FAQ Section -->
    <div class="post-section">
        <h2>❓ Frequently Asked Questions</h2>
        <div class="faq">
            <details>
                <summary>Is this APK safe to install on my phone?</summary>
                <p>Yes. We provide clean, verified package links. As a best practice, you can always scan downloaded APK files with Google Play Protect or your device's built-in security scanner.</p>
            </details>
            <details>
                <summary>Do I need root permissions to run this game?</summary>
                <p>No root access is required. The MOD works seamlessly on standard, non-rooted Android smartphones and tablets.</p>
            </details>
            <details>
                <summary>How do I update to the latest version without losing progress?</summary>
                <p>When a new update drops, download the updated APK from this page and install it directly over your existing installation without uninstalling first.</p>
            </details>
        </div>
    </div>

    <!-- Bottom Pre-Footer CTA Card -->
    <div class="bottom-cta">
        <h3>Download {app_name} MOD APK</h3>
        <p>Direct Download • Version {version}</p>
        {btn_group_html}
        <p class="disclaimer">
            Disclaimer: All files are provided for informational and testing purposes. All trademarks, logos, and brand names belong to their respective copyright holders.
        </p>
    </div>
</div>
"""
        return html_body.strip()
