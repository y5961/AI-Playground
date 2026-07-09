"""בדיקת סוכן השיחה עם RAG - אימות חיבור ל-LLM ותוכנות.

תסריט זה בודק שהסוכן עובד כנכון עם RAG וחיפוש סמנטי.
"""

from __future__ import annotations

import sys
from pathlib import Path

# הוספת src לנתיב כדי שנוכל לייבא מודולים מداخל src
sys.path.insert(0, str(Path(__file__).parent))

from agents.chat import answer
from core.store import get_store


def main():
    """הפעלת בדיקה של סוכן השיחה עם RAG."""
    print("=" * 70)
    print("בדיקת סוכן השיחה עם RAG - MVP Step 2")
    print("=" * 70)
    print()
    
    # קבלת ה-store
    store = get_store()
    
    # הוספת מקור לדוגמה
    print("📚 הוספת מקורות לדוגמה...")
    print("-" * 70)
    
    source1_content = """
    LangChain היא framework לבניית אפליקציות בעוצמת שפות הגדלות (LLMs).
    היא מספקת tools לטיפול בטקסט, טיפול בדיוקים, וחיבור למודלים שונים.
    LangChain משמשת בעיקר ליישומי RAG וסוכנים חכמים.
    """
    
    source1_id = store.add("source1", "LangChain Basics", source1_content)
    print(f"✓ הוסף מקור: {source1_id.name}")
    
    source2_content = """
    Retrieval-Augmented Generation (RAG) הוא טכניקה שמשלבת חיפוש מסמכים עם יצירת תוכן ב-LLM.
    התהליך:
    1. המשתמש שואל שאלה
    2. המערכת מחפשת מסמכים רלוונטיים
    3. המסמכים מועברים ל-LLM
    4. LLM משלב את המסמכים בתשובה
    
    יתרונות RAG:
    - תשובות מדויקות יותר בהתבסס על נתונים
    - ממנועות ידע דיוק גבוה
    - יכולת להתמודד עם מידע עדכני
    """
    
    source2_id = store.add("source2", "RAG Fundamentals", source2_content)
    print(f"✓ הוסף מקור: {source2_id.name}")
    print()
    
    # בדיקה 1: שאלה שהיא על אחד המקורות
    print("=" * 70)
    print("📝 בדיקה 1: שאלה על RAG")
    print("-" * 70)
    question1 = "מה זה RAG וכיצד זה עובד?"
    print(f"שאלה: {question1}")
    print()
    
    try:
        result1 = answer(question1, thread_id="test-1")
        print(f"תשובה:\n{result1.text}")
        print()
        if result1.sources:
            print(f"מקורות: {', '.join(result1.sources)}")
        print()
    except Exception as e:
        print(f"❌ שגיאה בבדיקה 1: {e}")
        print()
    
    # בדיקה 2: שאלה על LangChain
    print("=" * 70)
    print("📝 בדיקה 2: שאלה על LangChain")
    print("-" * 70)
    question2 = "למה משמשת LangChain?"
    print(f"שאלה: {question2}")
    print()
    
    try:
        result2 = answer(question2, thread_id="test-1")
        print(f"תשובה:\n{result2.text}")
        print()
        if result2.sources:
            print(f"מקורות: {', '.join(result2.sources)}")
        print()
    except Exception as e:
        print(f"❌ שגיאה בבדיקה 2: {e}")
        print()
    
    # בדיקה 3: בדיקת כלים - רשימת מקורות
    print("=" * 70)
    print("📝 בדיקה 3: שאלה שתשתמש בכלי list_sources")
    print("-" * 70)
    question3 = "אילו מקורות קיימים?"
    print(f"שאלה: {question3}")
    print()
    
    try:
        result3 = answer(question3, thread_id="test-1")
        print(f"תשובה:\n{result3.text}")
        print()
    except Exception as e:
        print(f"❌ שגיאה בבדיקה 3: {e}")
        print()
    
    # בדיקה 4: שאלה שלא קשורה למקורות
    print("=" * 70)
    print("📝 בדיקה 4: שאלה כללית (לא קשורה למקורות)")
    print("-" * 70)
    question4 = "מה הסדר ההופכי של המילה 'עברית'?"
    print(f"שאלה: {question4}")
    print()
    
    try:
        result4 = answer(question4, thread_id="test-2")
        print(f"תשובה:\n{result4.text}")
        print()
    except Exception as e:
        print(f"❌ שגיאה בבדיקה 4: {e}")
        print()
    
    # סיכום
    print("=" * 70)
    print("✅ הבדיקות הסתיימו!")
    print("=" * 70)
    print()
    print("הערות:")
    print("- אם ראית שגיאות בקשר ל-API keys, עדכן את הקובץ .env")
    print("- אם ראית שגיאות בקשר למקורות, בדוק שהם התווספו בהצלחה")


if __name__ == "__main__":
    main()
