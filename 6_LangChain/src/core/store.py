"""ניהול מאגר המקורות (Source Store) עם חיפוש סמנטי.

Step 2: RAG Implementation - ניהול מקורות בזיכרון עם וקטור סטור.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from core.sources import chunk_source, format_docs

# טעינת משתנים מסביבה מקובץ .env
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


@dataclass
class Source:
    """מקור (מסמך שהוטען על ידי המשתמש)."""

    id: str
    name: str
    content: str
    active: bool = True


class SourceStore:
    """
    מאגר המקורות - ניהול מסמכים, הטמעות וחיפוש סמנטי.
    
    כולל:
    - sources: רשימה של Source objects
    - vector_store: InMemoryVectorStore להטמעות וחיפוש סמנטי
    - embeddings: CohereEmbeddings להמרת טקסט לוקטורים
    """

    def __init__(self):
        """אתחול ה-store עם embeddings וקטור סטור."""
        self.sources: dict[str, Source] = {}
        self._embeddings = None  # Lazy loading
        self._vector_store = None  # Lazy loading
        self._source_chunks: dict[str, list[Document]] = {}
    
    @property
    def embeddings(self):
        """Lazy loading של OpenAIEmbeddings."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return self._embeddings
    
    @property
    def vector_store(self) -> InMemoryVectorStore:
        """Lazy loading של InMemoryVectorStore."""
        if self._vector_store is None:
            self._vector_store = InMemoryVectorStore(self.embeddings)
        return self._vector_store
    
    @vector_store.setter
    def vector_store(self, value: InMemoryVectorStore):
        """setter ל-vector_store."""
        self._vector_store = value

    def add(self, source_id: str, name: str, content: str) -> Source:
        """
        הוספת מקור חדש למאגר.
        
        תהליך:
        1. יצירת Source object
        2. חלוקת המקור לחתיכות (chunks)
        3. הטמעה והוספה ל-vector_store
        
        Args:
            source_id: מזהה ייחודי למקור
            name: שם המקור (קובץ, כותרת, וכו')
            content: תוכן המקור
        
        Returns:
            Source object שהתווסף
        """
        # יצירת Source object
        source = Source(id=source_id, name=name, content=content, active=True)
        self.sources[source_id] = source
        
        # חלוקת המקור לחתיכות
        chunks = chunk_source(source_id, name, content)
        self._source_chunks[source_id] = chunks
        
        # הוספה ל-vector store להטמעה וחיפוש
        self.vector_store.add_documents(chunks)
        
        return source

    def remove(self, source_id: str) -> bool:
        """
        הסרת מקור מהמאגר.
        
        הערה: הסרה מ-InMemoryVectorStore היא לא ישירה,
        אנחנו מבצעים rebuild של ה-vector store ללא החתיכות של המקור המוסר.
        
        Args:
            source_id: מזהה המקור להסרה
        
        Returns:
            True אם הוסר בהצלחה, False אם לא נמצא
        """
        if source_id not in self.sources:
            return False
        
        # הסרה מהמקורות
        del self.sources[source_id]
        
        # הסרה מהחתיכות
        if source_id in self._source_chunks:
            del self._source_chunks[source_id]
        
        # rebuild של vector store ללא החתיכות של המקור המוסר
        self._rebuild_vector_store()
        
        return True

    def search(self, query: str, k: int = 3) -> list[Document]:
        """
        חיפוש סמנטי במקורות הפעילים.
        
        Args:
            query: השאלה או הטקסט לחיפוש
            k: מספר התוצאות המובילות להחזרה
        
        Returns:
            רשימה של Document objects המתאימים לחיפוש
        """
        # חיפוש סמנטי דרך vector store
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def search_with_scores(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        """
        חיפוש סמנטי עם ציוני דמיון.
        
        Args:
            query: השאלה או הטקסט לחיפוש
            k: מספר התוצאות המובילות להחזרה
        
        Returns:
            רשימה של tuple (Document, score)
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return results

    def list(self) -> list[Source]:
        """
        החזרת כל המקורות (פעילים וכו').
        
        Returns:
            רשימה של Source objects
        """
        return list(self.sources.values())

    def list_active(self) -> list[Source]:
        """
        החזרת רק המקורות הפעילים.
        
        Returns:
            רשימה של Source objects פעילים
        """
        return [s for s in self.sources.values() if s.active]

    def get(self, source_id: str) -> Optional[Source]:
        """
        קבלת מקור ספציפי לפי ID.
        
        Args:
            source_id: מזהה המקור
        
        Returns:
            Source object או None אם לא נמצא
        """
        return self.sources.get(source_id)

    def set_active(self, source_id: str, active: bool) -> Optional[Source]:
        """
        הפעלה או ביטול הפעלה של מקור.
        
        Args:
            source_id: מזהה המקור
            active: האם להפעיל (True) או להשבית (False)
        
        Returns:
            Source object המעודכן, או None אם לא נמצא
        """
        source = self.sources.get(source_id)
        if source:
            source.active = active
        return source

    def active_ids(self) -> list[str]:
        """
        החזרת IDs של כל המקורות הפעילים.
        
        Returns:
            רשימה של source IDs פעילים
        """
        return [s.id for s in self.sources.values() if s.active]

    def _rebuild_vector_store(self):
        """
        סידור מחדש של vector store עם כל החתיכות הקיימות.
        
        משמש כשמוסרים מקורות, כדי להסיר את החתיכות שלהם מהחיפוש.
        """
        # יצירת vector store חדש
        self._vector_store = InMemoryVectorStore(self.embeddings)
        
        # הוספת כל החתיכות הקיימות
        all_chunks = []
        for chunks in self._source_chunks.values():
            all_chunks.extend(chunks)
        
        if all_chunks:
            self._vector_store.add_documents(all_chunks)


# -- Global Store Instance ------------------------------------------------

_store = SourceStore()


def get_store() -> SourceStore:
    """החזרת הinstance הגלובלי של SourceStore."""
    return _store
