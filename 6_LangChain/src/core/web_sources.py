"""ניהול מקורות מרשת ו-integration עם SourceStore.

Step 3: Web Search Integration - הוספת מקורות מחיפוש ברשת.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from core.store import SourceStore, get_store
from core.web_search import (
    FirecrawlClient,
    WebSearchResult,
    WebSourceBatch,
    create_search_queries,
    filter_quality_results,
)


@dataclass
class WebSource:
    """מקור מרשת עם metadata."""

    id: str
    title: str
    url: str
    content: str
    source_type: str = "web"  # "web" או "file"
    active: bool = True


class WebSourceManager:
    """ניהול מקורות מרשת עם Firecrawl integration."""

    def __init__(self, firecrawl_client: Optional[FirecrawlClient] = None):
        """
        אתחול מנהל המקורות.
        
        Args:
            firecrawl_client: Firecrawl client (או ייווצר אוטומטית)
        """
        self.firecrawl = firecrawl_client or FirecrawlClient()
        self.web_sources: dict[str, WebSource] = {}

    def search_and_add(
        self, topic: str, store: SourceStore, num_sources: int = 3
    ) -> list[WebSource]:
        """
        חיפוש בנושא והוספה של מקורות לstore.
        
        Args:
            topic: נושא החיפוש
            store: SourceStore instance
            num_sources: מספר מקורות להוסיף
        
        Returns:
            רשימה של WebSource שנוספו
        """
        added_sources = []

        # ביצוע חיפוש עמוק
        queries = create_search_queries(topic)
        all_results = []

        for query in queries:
            try:
                results = self.firecrawl.search(query, max_results=3)
                all_results.extend(results)
            except Exception as e:
                print(f"❌ שגיאה בחיפוש עבור '{query}': {e}")
                continue

        # סינון תוצאות איכותיות
        quality_results = filter_quality_results(all_results)

        # Scrape תוכן מ-top results
        for i, result in enumerate(quality_results[:num_sources]):
            try:
                # חילוץ תוכן מה-URL
                if not result.content:
                    result.content = self.firecrawl.scrape(result.url)

                # יצירת source ID ייחודי
                source_id = f"web_{uuid.uuid4().hex[:8]}"

                # הוספה ל-store
                if result.content:
                    store.add(
                        source_id=source_id,
                        name=f"🌐 {result.title}",
                        content=result.content,
                    )

                    # יצירת WebSource object
                    web_source = WebSource(
                        id=source_id,
                        title=result.title,
                        url=result.url,
                        content=result.content,
                        source_type="web",
                    )
                    self.web_sources[source_id] = web_source
                    added_sources.append(web_source)

                    print(f"✓ הוסף מקור: {result.title}")

            except Exception as e:
                print(f"❌ שגיאה בעיבוד {result.title}: {e}")
                continue

        return added_sources

    def get_web_source(self, source_id: str) -> Optional[WebSource]:
        """קבלת מקור רשת לפי ID."""
        return self.web_sources.get(source_id)

    def list_web_sources(self) -> list[WebSource]:
        """רשימה של כל מקורות הרשת."""
        return list(self.web_sources.values())


# -- Global instance --


_web_manager: Optional[WebSourceManager] = None


def get_web_manager() -> WebSourceManager:
    """קבלת מנהל מקורות הרשת הגלובלי."""
    global _web_manager
    if _web_manager is None:
        try:
            _web_manager = WebSourceManager()
        except ValueError as e:
            print(f"⚠️ אזהרה: {e}")
            print("💡 יש לקבוע את FIRECRAWL_API_KEY בקובץ .env")
            # יוצר dummy manager כדי להימנע מקריסה
            _web_manager = WebSourceManager.__new__(WebSourceManager)
            _web_manager.web_sources = {}
            _web_manager.firecrawl = None

    return _web_manager
