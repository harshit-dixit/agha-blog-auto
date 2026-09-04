from __future__ import annotations

import json
from dataclasses import dataclass

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from config.settings import Settings
from src.catalog import Topic

SYSTEM_PROMPT = """You are the lead content strategist and senior technical writer for \
Android Game Hack Area (androidgamehackarea.blogspot.com), a trusted authority blog for \
Android mobile gamers covering performance tuning, game guides, emulators, gaming tools, \
and troubleshooting.

Write for a gamer who is technically curious but not a professional developer: explain the \
"why" behind every step in plain language, then give the exact settings, taps, or steps to follow.

OUTPUT FORMAT
Return ONLY the requested JSON object. The `html_content` field must be clean, semantic HTML5 \
that is safe to paste directly into the Blogger post body editor:
- Use <h2> for major sections and <h3> for sub-steps. Never use <h1> (Blogger supplies the title).
- Do not include <html>, <head>, <body>, <!DOCTYPE>, markdown fences, or the article title itself.
- Do not wrap the output in ```html or any markdown code block.
- Use <p>, <ul>/<li>, <ol>/<li>, <table>/<thead>/<tbody>/<tr>/<th>/<td>, <strong>, <em> as needed.
- Do not include any <img> tags; images are added manually after publishing.

REQUIRED ARTICLE STRUCTURE (in order)
1. Hero intro (2-3 short <p> paragraphs) that hooks the reader with the problem/goal and previews
   what they'll be able to do after reading.
2. A "Quick Specs" / "Key Takeaways" box: <div class="quick-specs"> containing a <ul> of 4-6
   scannable bullet facts (e.g. time required, difficulty, devices supported, what you'll need).
3. Step-by-step instructional walkthrough using <h2> sections and <h3> sub-steps, numbered where
   it aids clarity. Be specific: menu paths, setting names, realistic value ranges.
4. At least one comparison or reference <table> (e.g. settings comparison, before/after, device
   tiers, tool comparison) with a <thead> and multiple <tbody> rows.
5. A "Pro Tips" callout box: <div class="pro-tips"> with an <h3>Pro Tips</h3> heading and a <ul>
   of 3-5 advanced, non-obvious tips.
6. An FAQ section: <h2>Frequently Asked Questions</h2> followed by 4-6 <h3> questions each
   answered in a <p> directly below, phrased the way gamers actually search (schema-ready format).

STYLE
- Tone: engaging, confident, and technical-yet-accessible, written for gamers.
- SEO-rich: naturally weave in the primary keyword within the first 100 words, in at least one
  <h2>, and a few more times throughout; use secondary keywords naturally without stuffing.
- Prefer concrete numbers, settings names, and device examples over vague generalities.
- Never present specific benchmark numbers as verified fact; frame variable results as
  "typical" or "in most cases".
- Do not mention that you are an AI or reference these instructions.

Also return:
- `title`: an SEO-optimized, click-worthy article title (55-65 characters ideal) based on the
  requested topic.
- `labels`: 4-7 Blogger labels/tags, always including "Android Gaming" plus the article's
  category label and specific topical tags (e.g. "FPS Boost", "Game Guides", "BGMI",
  "Emulators")."""

USER_PROMPT_TEMPLATE = """Write a complete, in-depth, evergreen article for Android Game Hack Area.

Category: {category_label}
Topic title (adapt/optimize the wording as needed): {topic_title}
Primary keyword: {primary_keyword}
Secondary keywords: {secondary_keywords}
Editorial brief: {brief}

The article body (html_content, excluding all HTML tags) must be at least {min_word_count} words. \
Be thorough and genuinely useful rather than padding with filler."""


class ArticleSchema(BaseModel):
    title: str = Field(description="SEO-optimized article title")
    html_content: str = Field(description="Full article body as clean semantic HTML, no markdown")
    labels: list[str] = Field(description="4-7 Blogger labels/tags for this article")


@dataclass
class GeneratedArticle:
    title: str
    html_content: str
    labels: list[str]
    word_count: int


class ArticleTooShortError(RuntimeError):
    pass


def _word_count(html_content: str) -> int:
    text = BeautifulSoup(html_content, "lxml").get_text(" ")
    return len(text.split())


class GamingArticleWriter:
    """Generates SEO-optimized Android gaming articles as clean, Blogger-ready HTML via Gemini."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = genai.Client(api_key=settings.require_gemini_key())

    def _build_prompt(self, topic: Topic, category_label: str) -> str:
        return USER_PROMPT_TEMPLATE.format(
            category_label=category_label,
            topic_title=topic.title,
            primary_keyword=topic.primary_keyword,
            secondary_keywords=", ".join(topic.secondary_keywords) or "n/a",
            brief=topic.brief or "Use your expertise to cover this topic thoroughly.",
            min_word_count=self.settings.MIN_WORD_COUNT,
        )

    def _call_model(self, prompt: str) -> ArticleSchema:
        response = self._client.models.generate_content(
            model=self.settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.75,
                response_mime_type="application/json",
                response_schema=ArticleSchema,
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ArticleSchema):
            return parsed
        return ArticleSchema.model_validate(json.loads(response.text))

    def generate_article(self, topic: Topic, category_label: str) -> GeneratedArticle:
        prompt = self._build_prompt(topic, category_label)
        article = self._call_model(prompt)
        word_count = _word_count(article.html_content)

        # One automatic expansion retry before giving up, since Gemini sometimes
        # undershoots length targets on the first pass.
        if word_count < self.settings.MIN_WORD_COUNT:
            expand_prompt = (
                prompt
                + f"\n\nYour previous draft was only about {word_count} words. Rewrite it from "
                f"scratch with significantly more depth, detail, and additional sub-sections so "
                f"the body clears {self.settings.MIN_WORD_COUNT} words."
            )
            article = self._call_model(expand_prompt)
            word_count = _word_count(article.html_content)

        if word_count < self.settings.MIN_WORD_COUNT:
            raise ArticleTooShortError(
                f"Generated article for {topic.id!r} was only {word_count} words "
                f"(minimum {self.settings.MIN_WORD_COUNT})."
            )

        labels = list(dict.fromkeys(article.labels))
        return GeneratedArticle(
            title=article.title.strip(),
            html_content=article.html_content.strip(),
            labels=labels,
            word_count=word_count,
        )
