"""פונקציות להנהלת מקורות וחלוקת טקסטים.

Step 2: RAG Implementation - טעינה וחלוקה של מקורות למקטעים.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_source(source_id: str, source_name: str, content: str) -> list[Document]:
    """
    חלוקת מקור לחתיכות קטנות יותר (chunks) להטמעה וחיפוש סמנטי.
    
    Args:
        source_id: מזהה הקוד
        source_name: שם הקוד
        content: תוכן הקוד
    
    Returns:
        רשימה של Document objects, כל אחד עם חתיכה מהקוד.
    """
    # אתחול ה-splitter עם פרמטרים סטנדרטיים
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    
    # חלוקת התוכן לחתיכות
    chunks = splitter.split_text(content)
    
    # יצירת Document objects עם metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source_id": source_id,
                "source_name": source_name,
            },
        )
        for chunk in chunks
    ]
    
    return documents


def format_docs(docs: list[Document]) -> str:
    """
    עיצוב מסמכים לצורה קריאה וידידותית ל-LLM.
    
    Args:
        docs: רשימה של Document objects מחיפוש סמנטי
    
    Returns:
        טקסט מעוצב כמו:
        [1] (source: intro.md)
        LangChain is a framework...
        
        [2] (source: rag_basics.pdf)
        Retrieval-Augmented Generation...
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, start=1):
        source_name = doc.metadata.get("source_name", "unknown")
        content = doc.page_content
        formatted_parts.append(f"[{i}] (source: {source_name})\n{content}")
    
    return "\n\n".join(formatted_parts)
