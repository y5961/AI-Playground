"""ניהול חיפוש ברשת ו-web scraping דרך Firecrawl.

Step 3: Web Search Integration - הוספת מקורות מהרשת דרך Firecrawl.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

# טעינת משתנים מסביבה
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(_env_path)


@dataclass
class WebSearchResult:
    """תוצאה בודדת מחיפוש ברשת."""

    title: str
    url: str
    description: str
    content: Optional[str] = None


@dataclass
class WebSourceBatch:
    """קבוצה של מקורות מרשת לעיבוד."""

    search_query: str
    results: list[WebSearchResult]


class FirecrawlClient:
    """קליינט ל-Firecrawl API - חיפוש וscraping של דפים."""

    def __init__(self, api_key: Optional[str] = None):
        """
        אתחול קליינט Firecrawl.
        
        Args:
            api_key: Firecrawl API key (או מ-FIRECRAWL_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key not found. "
                "Set FIRECRAWL_API_KEY environment variable or pass api_key parameter."
            )
        self.base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        """
        חיפוש ברשת דרך Firecrawl.
        
        Args:
            query: שאילתת החיפוש
            max_results: מספר התוצאות המקסימלי
        
        Returns:
            רשימה של WebSearchResult objects
        """
        endpoint = f"{self.base_url}/search"
        payload = {
            "query": query,
            "limit": max_results,
            "scrapeOptions": {
                "formats": ["markdown"],
            },
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                result = WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", ""),
                    content=item.get("content"),
                )
                results.append(result)

            return results

        except requests.exceptions.RequestException as e:
            raise Exception(f"Firecrawl search failed: {e}")

    def scrape(self, url: str) -> str:
        """
        Scrape תוכן מ-URL יחיד.
        
        Args:
            url: ה-URL לscrape
        
        Returns:
            התוכן שחולץ (Markdown format)
        """
        endpoint = f"{self.base_url}/scrape"
        payload = {
            "url": url,
            "formats": ["markdown"],
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            return data.get("markdown", "")

        except requests.exceptions.RequestException as e:
            raise Exception(f"Firecrawl scrape failed for {url}: {e}")

    def crawl(self, url: str, max_pages: int = 5) -> list[str]:
        """
        Crawl אתר שלם וחילוץ תוכן מכל הדפים.
        
        Args:
            url: URL של האתר לcrawl
            max_pages: מספר הדפים המקסימלי
        
        Returns:
            רשימה של תוכנים (Markdown format)
        """
        endpoint = f"{self.base_url}/crawl"
        payload = {
            "url": url,
            "limit": max_pages,
            "scrapeOptions": {
                "formats": ["markdown"],
            },
        }

        try:
            response = requests.post(endpoint, json=payload, headers=self.headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            contents = []
            for result in data.get("results", []):
                if "markdown" in result:
                    contents.append(result["markdown"])

            return contents

        except requests.exceptions.RequestException as e:
            raise Exception(f"Firecrawl crawl failed for {url}: {e}")


def create_search_queries(topic: str) -> list[str]:
    """
    יצירת מספר שאילתות חיפוש עבור נושא כדי לכסות אותו מכמה זוויות.
    
    Args:
        topic: הנושא לחיפוש
    
    Returns:
        רשימה של שאילתות חיפוש
    """
    queries = [
        f"{topic}",
        f"מה זה {topic}",
        f"{topic} טיפים ודוגמאות",
        f"{topic} טוטוריאל מבוא",
        f"{topic} מדריך שלם",
    ]
    return queries


def filter_quality_results(
    results: list[WebSearchResult], min_description_length: int = 50
) -> list[WebSearchResult]:
    """
    סינון תוצאות לפי איכות.
    
    Args:
        results: רשימה של תוצאות חיפוש
        min_description_length: אורך תיאור מינימלי
    
    Returns:
        תוצאות מסוננות
    """
    filtered = []
    seen_urls = set()

    for result in results:
        # תנאים לאיכות גבוהה
        if result.url in seen_urls:
            continue  # תוצאה דופליקט
        if len(result.description) < min_description_length:
            continue  # תיאור קצר מדי
        if not result.title:
            continue  # בלי כותרת

        seen_urls.add(result.url)
        filtered.append(result)

    return filtered


async def perform_deep_search(topic: str, client: FirecrawlClient) -> WebSourceBatch:
    """
    ביצוע חיפוש עמוק - מספר שאילתות וscraping של תוצאות.
    
    Args:
        topic: הנושא לחיפוש
        client: Firecrawl client
    
    Returns:
        WebSourceBatch עם תוצאות
    """
    queries = create_search_queries(topic)
    all_results = []

    for query in queries:
        try:
            results = client.search(query, max_results=3)
            all_results.extend(results)
        except Exception as e:
            print(f"שגיאה בחיפוש עבור '{query}': {e}")
            continue

    # סינון תוצאות איכותיות
    quality_results = filter_quality_results(all_results)

    # Scrape תוכן מהתוצאות המובילות
    for result in quality_results[:3]:  # Top 3 תוצאות
        try:
            content = client.scrape(result.url)
            result.content = content
        except Exception as e:
            print(f"שגיאה בscraping של {result.url}: {e}")

    return WebSourceBatch(search_query=topic, results=quality_results)
