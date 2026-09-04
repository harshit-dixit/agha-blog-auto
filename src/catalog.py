from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class Topic(BaseModel):
    id: str
    title: str
    primary_keyword: str
    secondary_keywords: list[str] = []
    brief: str = ""


class Category(BaseModel):
    id: str
    label: str
    description: str
    topics: list[Topic]


class TopicCatalog(BaseModel):
    categories: dict[str, Category]

    def all_topics(self) -> list[tuple[str, Topic]]:
        return [
            (category_id, topic)
            for category_id, category in self.categories.items()
            for topic in category.topics
        ]

    def get_topic(self, topic_id: str) -> tuple[str, Topic] | None:
        for category_id, topic in self.all_topics():
            if topic.id == topic_id:
                return category_id, topic
        return None

    def get_category(self, category_id: str) -> Category | None:
        return self.categories.get(category_id)


def load_catalog(path: Path) -> TopicCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = {
        category_id: Category(id=category_id, **category_data)
        for category_id, category_data in raw["categories"].items()
    }
    return TopicCatalog(categories=categories)
