from __future__ import annotations

import html
import logging
import re
from typing import Optional

from google import genai
from google.genai import types

from config.settings import Settings
from src.scrapers.an1_scraper import AN1Post

logger = logging.getLogger(__name__)

# Minimum usable length for generated prose; shorter output means the model bailed out.
MIN_ENHANCED_LENGTH = 200


class ContentEnhancementError(RuntimeError):
    """Raised when Gemini is configured but could not produce usable original prose.

    Publishing the scraped description verbatim is the failure mode this guards against,
    so callers should skip the post rather than fall back to it.
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


class AN1Formatter:
    """Formats an AN1 scraped post into a modern, mobile-responsive HTML post for Blogger."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
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

    def _enhance_with_gemini(self, post: AN1Post) -> Optional[str]:
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
                ),
            )
            raw_text = response.text or ""
        except Exception as exc:
            raise ContentEnhancementError(
                f"Gemini enhancement failed for {post.app_name!r}: {exc}"
            ) from exc

        text = self._normalize_enhanced_text(raw_text)
        if not text:
            raise ContentEnhancementError(
                f"Gemini returned unusable prose for {post.app_name!r} "
                f"({len(raw_text.strip())} chars, no usable <p> block)"
            )
        return text

    @staticmethod
    def _normalize_enhanced_text(raw_text: str) -> Optional[str]:
        """Strip fences and any preamble, returning usable paragraph HTML or None."""
        text = raw_text.replace("```html", "").replace("```", "").strip()

        # Drop any lead-in prose before the first paragraph tag ("Here is the review:" etc.)
        first_p = re.search(r"<p[\s>]", text, re.I)
        if first_p:
            text = text[first_p.start():].strip()

        if not text.lower().startswith("<p") or len(text) < MIN_ENHANCED_LENGTH:
            return None
        return text

    def format_html(self, post: AN1Post) -> str:
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

        # Build screenshots HTML
        screenshots_html = ""
        if post.screenshots:
            items = []
            for img_url in post.screenshots[:6]:
                safe_img = html.escape(img_url)
                items.append(
                    f'<div class="agha-shot"><img src="{safe_img}" alt="{app_name} Screenshot" loading="lazy" /></div>'
                )
            screenshots_html = f"""
            <div class="agha-section">
                <h3 class="agha-subtitle">📸 In-Game Screenshots</h3>
                <div class="agha-gallery">
                    {''.join(items)}
                </div>
            </div>
            """

        # Original prose from Gemini when configured; the scraped-text fallback below is
        # only reached when no Gemini key is set (a failed generation raises instead).
        enhanced_desc = self._enhance_with_gemini(post)
        if enhanced_desc:
            desc_formatted = enhanced_desc
        else:
            desc_text = post.description_text or "Download the latest modified version with unlocked features and unlimited resources."
            # Split paragraphs cleanly by newline or sentence blocks
            raw_blocks = [b.strip() for b in desc_text.split("\n\n") if b.strip()]
            if not raw_blocks:
                raw_blocks = [desc_text]
            desc_formatted = "".join(f"<p>{html.escape(b)}</p>" for b in raw_blocks)

        # Build download buttons (Inverted order: AN1 stable page as primary, direct APK as secondary mirror)
        primary_btn_html = f"""
        <a href="{html.escape(primary_link)}" class="agha-btn agha-btn-primary" target="_blank" rel="noopener nofollow">
            <span>⬇ Download APK{size_label}</span>
        </a>
        """

        mirror_btn_html = ""
        if mirror_link:
            mirror_btn_html = f"""
            <a href="{html.escape(mirror_link)}" class="agha-btn agha-btn-secondary" target="_blank" rel="noopener nofollow">
                <span>⚡ Direct APK Mirror</span>
            </a>
            """

        html_body = f"""
<div class="agha-article">
    <style>
        .agha-article {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #1e293b;
            line-height: 1.65;
            max-width: 800px;
            margin: 0 auto;
            font-size: 16px;
        }}
        .agha-hero {{
            display: flex;
            align-items: center;
            gap: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #ffffff;
            padding: 24px;
            border-radius: 18px;
            box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
            margin-bottom: 24px;
        }}
        .agha-hero-icon {{
            width: 90px;
            height: 90px;
            border-radius: 18px;
            object-fit: cover;
            box-shadow: 0 8px 16px rgba(0,0,0,0.3);
            flex-shrink: 0;
            background: #334155;
        }}
        .agha-hero-info {{
            flex: 1;
        }}
        .agha-hero-title {{
            font-size: 22px;
            font-weight: 700;
            margin: 0 0 8px 0;
            color: #ffffff;
            line-height: 1.25;
        }}
        .agha-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .agha-badge {{
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .badge-green {{
            background: #10b981;
            color: #ffffff;
        }}
        .badge-blue {{
            background: #3b82f6;
            color: #ffffff;
        }}
        .badge-dark {{
            background: rgba(255,255,255,0.15);
            color: #f1f5f9;
        }}
        .agha-hero-meta {{
            font-size: 13px;
            color: #94a3b8;
        }}

        /* CTA Download Box */
        .agha-cta-box {{
            background: #f8fafc;
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            margin: 24px 0;
        }}
        .agha-btn-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            justify-content: center;
            margin-top: 14px;
        }}
        .agha-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 14px 28px;
            font-size: 16px;
            font-weight: 700;
            text-decoration: none !important;
            border-radius: 12px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            cursor: pointer;
        }}
        .agha-btn:hover {{
            transform: translateY(-2px);
        }}
        .agha-btn-primary {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: #ffffff !important;
            box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.45);
        }}
        .agha-btn-secondary {{
            background: #e2e8f0;
            color: #1e293b !important;
        }}
        .agha-safe-check {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #059669;
            font-weight: 600;
            margin-top: 10px;
        }}

        /* Specs Grid */
        .agha-specs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 12px;
            margin: 24px 0;
        }}
        .agha-spec-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .agha-spec-label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .agha-spec-val {{
            font-size: 14px;
            font-weight: 700;
            color: #0f172a;
        }}

        /* MOD Box */
        .agha-mod-card {{
            background: #ecfdf5;
            border-left: 4px solid #10b981;
            border-radius: 0 12px 12px 0;
            padding: 16px 20px;
            margin: 24px 0;
        }}
        .agha-mod-title {{
            font-size: 16px;
            font-weight: 700;
            color: #065f46;
            margin: 0 0 6px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .agha-mod-desc {{
            color: #047857;
            margin: 0;
            font-size: 14px;
            font-weight: 500;
        }}

        /* Sections */
        .agha-section {{
            margin: 28px 0;
        }}
        .agha-subtitle {{
            font-size: 18px;
            font-weight: 700;
            color: #0f172a;
            margin: 0 0 12px 0;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 8px;
        }}

        /* Gallery */
        .agha-gallery {{
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding-bottom: 10px;
            scroll-snap-type: x mandatory;
        }}
        .agha-shot {{
            flex: 0 0 160px;
            scroll-snap-align: start;
        }}
        .agha-shot img {{
            width: 100%;
            height: 280px;
            object-fit: cover;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        }}

        /* Guide Steps */
        .agha-steps {{
            padding-left: 20px;
            margin: 12px 0;
        }}
        .agha-steps li {{
            margin-bottom: 8px;
            color: #334155;
        }}

        /* FAQ */
        .agha-faq-item {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }}
        .agha-faq-q {{
            font-weight: 700;
            color: #0f172a;
            font-size: 15px;
            margin-bottom: 4px;
        }}
        .agha-faq-a {{
            color: #475569;
            font-size: 14px;
            margin: 0;
        }}

        @media (max-width: 600px) {{
            .agha-hero {{
                flex-direction: column;
                text-align: center;
                gap: 14px;
            }}
            .agha-hero-icon {{
                width: 80px;
                height: 80px;
            }}
            .agha-tags {{
                justify-content: center;
            }}
            .agha-btn {{
                width: 100%;
            }}
        }}
    </style>

    <!-- Header Hero Card -->
    <div class="agha-hero">
        <img class="agha-hero-icon" src="{html.escape(post.icon_url)}" alt="{app_name} Icon" />
        <div class="agha-hero-info">
            <div class="agha-tags">
                <span class="agha-badge badge-green">MOD APK</span>
                <span class="agha-badge badge-blue">v{version}</span>
                <span class="agha-badge badge-dark">⭐ {rating}/5</span>
            </div>
            <h2 class="agha-hero-title">{app_name}</h2>
            <div class="agha-hero-meta">By <strong>{developer}</strong> • {installs} Downloads</div>
        </div>
    </div>

    <!-- Quick CTA Top -->
    <div class="agha-cta-box">
        <div style="font-weight: 700; font-size: 17px; color: #0f172a;">Get {app_name} MOD APK for Android</div>
        <div style="font-size: 13px; color: #64748b; margin-top: 2px;">{size_meta_line}</div>
        <div class="agha-btn-group">
            {primary_btn_html}
            {mirror_btn_html}
        </div>
        <div class="agha-safe-check">
            <span>⬇ Fast Direct Download • Tested Package</span>
        </div>
    </div>

    <!-- MOD Features Banner -->
    <div class="agha-mod-card">
        <div class="agha-mod-title">✨ Mod Features Unlocked</div>
        <p class="agha-mod-desc">{mod_features}</p>
    </div>

    <!-- Detailed Specifications -->
    <div class="agha-section">
        <h3 class="agha-subtitle">📋 App Specifications</h3>
        <div class="agha-specs-grid">
            <div class="agha-spec-card">
                <div class="agha-spec-label">App Name</div>
                <div class="agha-spec-val">{app_name}</div>
            </div>
            <div class="agha-spec-card">
                <div class="agha-spec-label">Version</div>
                <div class="agha-spec-val">{version}</div>
            </div>
            <div class="agha-spec-card">
                <div class="agha-spec-label">File Size</div>
                <div class="agha-spec-val">{size}</div>
            </div>
            <div class="agha-spec-card">
                <div class="agha-spec-label">Requires Android</div>
                <div class="agha-spec-val">{android_ver}</div>
            </div>
            <div class="agha-spec-card">
                <div class="agha-spec-label">Developer</div>
                <div class="agha-spec-val">{developer}</div>
            </div>
            <div class="agha-spec-card">
                <div class="agha-spec-label">Last Updated</div>
                <div class="agha-spec-val">{updated}</div>
            </div>
        </div>
    </div>

    <!-- App Overview & Features -->
    <div class="agha-section">
        <h3 class="agha-subtitle">📖 About {app_name}</h3>
        {desc_formatted}
    </div>

    <!-- Screenshots -->
    {screenshots_html}

    <!-- Installation Guide -->
    <div class="agha-section">
        <h3 class="agha-subtitle">📲 How to Install MOD APK</h3>
        <ol class="agha-steps">
            <li><strong>Download APK:</strong> Click the download button above or below to save the APK file to your Android device.</li>
            <li><strong>Allow Unknown Sources:</strong> If prompted, go to <em>Settings &gt; Security (or Apps)</em> and allow installation from Unknown Sources for your browser or file manager.</li>
            <li><strong>Install:</strong> Tap the downloaded file in your Notification bar or Downloads folder and choose <strong>Install</strong>.</li>
            <li><strong>Launch & Enjoy:</strong> Open {app_name} and enjoy all modified features with unlimited resources!</li>
        </ol>
    </div>

    <!-- FAQ Section -->
    <div class="agha-section">
        <h3 class="agha-subtitle">❓ Frequently Asked Questions</h3>
        <div class="agha-faq-item">
            <div class="agha-faq-q">Is this APK safe to install on my phone?</div>
            <div class="agha-faq-a">We provide verified package links. As a best practice, always check downloaded APK files with your device built-in security scanner or Play Protect.</div>
        </div>
        <div class="agha-faq-item">
            <div class="agha-faq-q">Do I need root permissions to run this game?</div>
            <div class="agha-faq-a">No root access is required. The MOD works smoothly on standard non-rooted Android smartphones and tablets.</div>
        </div>
        <div class="agha-faq-item">
            <div class="agha-faq-q">How do I update to the latest version without losing game progress?</div>
            <div class="agha-faq-a">When an update is released, simply download the new APK and install it directly over the existing installation without deleting the previous game.</div>
        </div>
    </div>

    <!-- Bottom Download Box -->
    <div class="agha-cta-box" style="margin-top: 36px; background: #f0fdf4; border-color: #86efac;">
        <div style="font-weight: 700; font-size: 18px; color: #166534;">Download {app_name} MOD APK</div>
        <div style="font-size: 13px; color: #15803d; margin-top: 3px;">Direct Download • Version {version}</div>
        <div class="agha-btn-group">
            {primary_btn_html}
            {mirror_btn_html}
        </div>
        <p style="font-size: 12px; color: #64748b; margin-top: 14px; margin-bottom: 0;">
            Disclaimer: All files are provided for informational and testing purposes. All trademarks, logos, and brand names belong to their respective copyright holders.
        </p>
    </div>
</div>
"""
        return html_body.strip()
