"""וכיל הצ'אט: סוכן שיחה עם RAG ותוכנות.

MVP - Step 2: סוכן שיחה שמשתמש במקורות דרך חיפוש סמנטי (RAG).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from core.sources import format_docs
from core.store import SourceStore, get_store


# טעינת משתנים מסביבה מקובץ .env
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path)


# -- Answer Object - תוצאת השיחה --------------------------------------------------


@dataclass
class Answer:
    """תוצאה של שיחה עם ה-LLM."""

    text: str
    """התשובה מה-LLM."""

    sources: list[str]
    """רשימה של מקורות שנעשה בהם שימוש בתשובה."""


# -- State Definition -------------------------------------------------------


class ChatState(TypedDict):
    """מצב השיחה עבור גרף LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]


# -- System Prompt ----------------------------------------------------------

SYSTEM_PROMPT = """אתה עוזר חכם בשם NotebookLM.
יש לך גישה למסמכים שהועלו וכלים לחיפוש בהם.
תפקידך להשיב על שאלות המשתמש בצורה ברורה וקצרה.

כלים שלך:
- search_sources: חיפוש סמנטי במקורות הקיימים. השתמש בזה כשהמשתמש שואל שאלות על תוכן הקבצים.
- list_sources: רשימה של כל המקורות הקיימים
- get_source: קבלת מקור מלא לפי ID

הנחיות:
1. עבור שאלות הקשורות למסמכים:
   - חפש תחילה בחיפוש סמנטי (search_sources)
   - השתמש תמיד במידע שנשלה וציין מקורות בתשובה
   - אם אין מקורות קיימים, אמור למשתמש שצריך להטעין קבצים תחילה

2. עבור שאלות ידע כללי שאינן קשורות כלל למסמכים (כמו איות, מתמטיקה או טריוויה כללית):
   - מותר לך לענות ישירות מהידע הכללי שלך
   - הבהר למשתמש שהמידע אינו מבוסס על המקורות שהועלו
   - אם אפשר, חלוק עם המקורות הקיימים

3. לשאלות מעורבות - שלב בין המידע מהמסמכים והידע הכללי, עם הפרדה ברורה בין המקורות"""


# -- Tools - כלים לשימוש בסוכן -----------------------------------------------


def make_tools(store: SourceStore) -> list[Any]:
    """
    יצירת רשימה של כלים (tools) עבור הסוכן.
    
    Args:
        store: SourceStore instance
    
    Returns:
        רשימה של פונקציות tools המוגדרות עם @tool decorator
    """
    
    @tool
    def search_sources(query: str) -> str:
        """
        חיפוש סמנטי במקורות הקיימים.
        
        השתמש בזה כדי למצוא מידע רלוונטי בקבצים המטופלים.
        
        Args:
            query: השאלה או הטקסט לחיפוש
        
        Returns:
            טקסט מעוצב עם התוצאות, כמו:
            [1] (source: intro.md)
            LangChain is a framework...
        """
        # חיפוש סמנטי
        results = store.search(query, k=3)
        
        if not results:
            return "לא נמצאו תוצאות רלוונטיות במקורות הקיימים."
        
        # עיצוב התוצאות
        formatted = format_docs(results)
        return formatted
    
    @tool
    def list_sources() -> str:
        """
        קבלת רשימה של כל המקורות הקיימים.
        
        Returns:
            טקסט עם רשימת המקורות והערות על הם
        """
        sources = store.list()
        
        if not sources:
            return "אין מקורות קיימים. יש להטעין קבצים תחילה."
        
        # יצירת רשימה מעוצבת
        lines = ["מקורות קיימים:"]
        for i, source in enumerate(sources, 1):
            status = "✓ פעיל" if source.active else "✗ לא פעיל"
            chars = len(source.content)
            lines.append(f"  {i}. {source.name} ({chars} תווים) [{status}]")
        
        return "\n".join(lines)
    
    @tool
    def get_source(source_id: str) -> str:
        """
        קבלת מקור מלא לפי ID.
        
        Args:
            source_id: מזהה המקור
        
        Returns:
            תוכן המקור או הודעת שגיאה
        """
        source = store.get(source_id)
        
        if not source:
            return f"לא נמצא מקור עם ID: {source_id}"
        
        return f"# {source.name}\n\n{source.content}"
    
    return [search_sources, list_sources, get_source]


# -- Agent Setup with Tools -------------------------------------------------


def _build_agent():
    """בונה את סוכן השיחה עם כלים ו-RAG."""
    
    # קבלת ה-store הגלובלי
    store = get_store()
    
    # יצירת הכלים
    tools = make_tools(store)
    
    # הגדרת מודל OpenAI
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # חיבור הכלים למודל
    model_with_tools = model.bind_tools(tools)
    
    # יצירת גרף המצב
    workflow = StateGraph(ChatState)
    
    def chat_node(state: ChatState) -> ChatState:
        """צומת שיחה: שולח את ההודעות למודל עם כלים."""
        
        # הוספת prompt המערכת כהודעה ראשונה
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        messages.extend(state["messages"])
        
        # קריאה למודל עם כלים
        response = model_with_tools.invoke(messages)
        
        # החזרת המצב המעודכן עם התשובה
        return {"messages": [response]}
    
    # צומת לביצוע כלים
    tool_node = ToolNode(tools)
    
    def route_tools(state: ChatState) -> str:
        """בחירה בין צומת כלים לסוף על פי התוצאה של המודל."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # אם יש tool_calls, בצע כלים
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # אחרת, סיים
        return "end"
    
    # הוספת צומות לגרף
    workflow.add_node("chat", chat_node)
    workflow.add_node("tools", tool_node)
    
    # הגדרת הקצוות
    workflow.add_edge(START, "chat")
    workflow.add_conditional_edges(
        "chat",
        route_tools,
        {"tools": "tools", "end": END},
    )
    workflow.add_edge("tools", "chat")
    
    # הידור הגרף עם checkpointer לשמירת היסטוריה
    checkpointer = MemorySaver()
    agent = workflow.compile(checkpointer=checkpointer)
    
    return agent


# -- Global Agent Instance --------------------------------------------------

_agent = None


def _get_agent():
    """מחזירה את הסוכן, בונה אותו אם לא קיים עדיין."""
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


# -- Public API - אנטרפייס ציבורי ----------------------------------------


def answer(message: str, thread_id: str = "default") -> Answer:
    """
    שולח הודעה לסוכן השיחה ומקבל תשובה מבוססת על מקורות.
    
    Args:
        message: הודעת המשתמש.
        thread_id: מזהה ה-thread לשמירת היסטוריה של השיחה.
    
    Returns:
        Answer: אובייקט עם הטקסט של התשובה ורשימת המקורות.
    """
    agent = _get_agent()
    
    # הכנת input עם הודעת המשתמש
    input_state = {"messages": [HumanMessage(content=message)]}
    
    # הפעלת הסוכן עם שמירת היסטוריה
    config = {"configurable": {"thread_id": thread_id}}
    output = agent.invoke(input_state, config=config)
    
    # חילוץ התשובה של המודל (ההודעה האחרונה שאינה tool message)
    messages = output["messages"]
    response_text = ""
    sources = []
    
    # חיפוש התשובה הסופית (ההודעה האחרונה של המודל שאינה tool message)
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not isinstance(msg, ToolMessage):
            response_text = msg.content
            break
    
    # זיהוי מקורות שנעשה בהם שימוש (בחילוץ metadata מ-tool messages)
    for msg in messages:
        if isinstance(msg, ToolMessage) and hasattr(msg, "artifact"):
            # אם tool message יש artifact, זה כנראה source
            pass
    
    # החזרת Answer object
    return Answer(text=response_text, sources=sources)
