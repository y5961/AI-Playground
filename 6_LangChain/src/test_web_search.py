"""בדיקת חיפוש ברשת - Step 3 Web Search Integration.

תסריט זה בוחן את היכולת לחפש ברשת הוסיף מקורות למאגר.
"""

from __future__ import annotations

import sys
from pathlib import Path

# הוספת src לנתיב
sys.path.insert(0, str(Path(__file__).parent))

from api.schemas import WebSearchRequest
from api.services import search_web
from core.store import get_store


def main():
    """הפעלת בדיקה של חיפוש ברשת."""
    print("=" * 70)
    print("בדיקת חיפוש ברשת - MVP Step 3")
    print("=" * 70)
    print()

    store = get_store()

    # בדיקה 1: חיפוש בנושא
    print("📚 בדיקה 1: חיפוש בנושא 'LangChain'")
    print("-" * 70)

    req = WebSearchRequest(topic="LangChain", num_sources=2)

    try:
        result = search_web(req)
        print(f"✓ נושא: {result.topic}")
        print(f"✓ מקורות שנוספו: {result.sources_added}")
        print(f"✓ סטטוס: {result.status}")
        print()

        if result.source_ids:
            print("מקורות שנוספו:")
            for source_id in result.source_ids:
                source = store.get(source_id)
                if source:
                    chars = len(source.content)
                    print(f"  - {source.name} ({chars} תווים)")
            print()

    except Exception as e:
        print(f"❌ שגיאה בבדיקה 1: {e}")
        print()

    # בדיקה 2: חיפוש בנושא אחר
    print("=" * 70)
    print("📚 בדיקה 2: חיפוש בנושא 'RAG Retrieval Augmented Generation'")
    print("-" * 70)

    req2 = WebSearchRequest(topic="RAG Retrieval Augmented Generation", num_sources=2)

    try:
        result2 = search_web(req2)
        print(f"✓ נושא: {result2.topic}")
        print(f"✓ מקורות שנוספו: {result2.sources_added}")
        print(f"✓ סטטוס: {result2.status}")
        print()

    except Exception as e:
        print(f"❌ שגיאה בבדיקה 2: {e}")
        print()

    # בדיקה 3: רשימת כל המקורות
    print("=" * 70)
    print("📚 בדיקה 3: רשימת כל המקורות")
    print("-" * 70)

    all_sources = store.list()
    print(f"סה\"כ מקורות: {len(all_sources)}")
    for source in all_sources:
        status = "✓ פעיל" if source.active else "✗ לא פעיל"
        chars = len(source.content)
        print(f"  - {source.name} ({chars} תווים) [{status}]")
    print()

    print("=" * 70)
    print("✅ בדיקות הסתיימו!")
    print("=" * 70)
    print()
    print("💡 הערות:")
    print("- אם ראית שגיאה 'Firecrawl API key not configured':")
    print("  1. עבור ל-https://app.firecrawl.dev")
    print("  2. יצור חשבון חדש")
    print("  3. העתק את ה-API key")
    print("  4. הדבק בקובץ .env בשורה FIRECRAWL_API_KEY")
    print("- אם הכל עבד, המקורות מהרשת התווספו למאגר! 🎉")


if __name__ == "__main__":
    main()
