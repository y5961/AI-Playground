"""Backend services: translate API requests into the LangChain code.

This is the thin layer between the stable API contract and the per-stage implementations.
Source management and notes are product glue (in-memory); chat is powered by the stage-2
agent; Studio artifacts raise ``ComingSoon`` until structured output lands in stage 3.
"""

from __future__ import annotations

import uuid

from api.schemas import (
    ArtifactKind,
    ChatRequest,
    ChatResponse,
    Citation,
    Note,
    SourceDetail,
    SourceInfo,
    WebSearchRequest,
    WebSearchResponse,
)
from agents import chat
from core.store import get_store
from core.web_sources import get_web_manager


class ComingSoon(Exception):
    """Raised for a product capability that is advertised but not wired up yet."""


# -- sources -------------------------------------------------------------------


def _to_info(source) -> SourceInfo:
    return SourceInfo(
        id=source.id, name=source.name, chars=len(source.content), active=source.active
    )


def list_sources() -> list[SourceInfo]:
    store = get_store()
    return [_to_info(s) for s in store.list()]


def get_source(source_id: str) -> SourceDetail | None:
    store = get_store()
    source = store.get(source_id)
    if source is None:
        return None
    return SourceDetail(
        id=source.id,
        name=source.name,
        chars=len(source.content),
        active=source.active,
        content=source.content,
    )


def add_source(name: str | None, content: str) -> SourceInfo:
    name = (name or "").strip() or _auto_name(content)
    source_id = uuid.uuid4().hex[:8]
    store = get_store()
    return _to_info(store.add(source_id=source_id, name=name, content=content))


def set_source_active(source_id: str, active: bool) -> SourceInfo | None:
    store = get_store()
    source = store.set_active(source_id, active)
    return _to_info(source) if source else None


def remove_source(source_id: str) -> bool:
    store = get_store()
    return store.remove(source_id)


def _auto_name(content: str) -> str:
    first_line = content.strip().splitlines()[0] if content.strip() else "Pasted source"
    first_line = first_line.lstrip("# ").strip()
    return (first_line[:40] or "Pasted source") + ".txt"


# -- web search ----------------------------------------------------------------


def search_web(req: WebSearchRequest) -> WebSearchResponse:
    """
    חיפוש ברשת והוספת מקורות למאגר.
    
    Args:
        req: WebSearchRequest עם נושא ומספר מקורות
    
    Returns:
        WebSearchResponse עם תוצאות ו-IDs של מקורות שנוספו
    """
    try:
        store = get_store()
        web_manager = get_web_manager()

        if not web_manager.firecrawl:
            return WebSearchResponse(
                topic=req.topic,
                sources_added=0,
                source_ids=[],
                status="❌ Firecrawl API key not configured",
            )

        # חיפוש והוספה
        added = web_manager.search_and_add(req.topic, store, req.num_sources)

        return WebSearchResponse(
            topic=req.topic,
            sources_added=len(added),
            source_ids=[s.id for s in added],
            status=f"✓ הוסף {len(added)} מקורות",
        )

    except Exception as e:
        return WebSearchResponse(
            topic=req.topic,
            sources_added=0,
            source_ids=[],
            status=f"❌ שגיאה: {str(e)}",
        )


# -- chat ----------------------------------------------------------------------


def run_chat(req: ChatRequest) -> ChatResponse:
    """Answer a chat turn with the stage-2 agent, grounded in the active sources."""
    store = get_store()
    if not store.active_ids():
        return ChatResponse(
            answer="No active sources. Enable at least one source on the left to chat.",
            engine="chat",
        )

    result = chat.answer(req.message, thread_id=req.thread_id or "default")
    citations = [Citation(source=name) for name in result.sources]
    return ChatResponse(answer=result.text, citations=citations, engine="chat")


# -- studio (artifacts) --------------------------------------------------------

ARTIFACTS: list[ArtifactKind] = [
    ArtifactKind(key="infographic", title="Infographic", icon="📊", status="planned"),
    ArtifactKind(key="powerpoint", title="PowerPoint", icon="📑", status="planned"),
    ArtifactKind(key="summary", title="Summary", icon="📄", status="planned"),
    ArtifactKind(key="faq", title="FAQ", icon="❓", status="planned"),
]

_ARTIFACTS_BY_KEY = {a.key: a for a in ARTIFACTS}


def list_artifacts() -> list[ArtifactKind]:
    return ARTIFACTS


def generate_artifact(kind: str, impl: str) -> Note:
    # Artifact generation lands with structured output; until then it's "coming soon".
    artifact = _ARTIFACTS_BY_KEY.get(kind)
    title = artifact.title if artifact else "This artifact"
    raise ComingSoon(f"{title} generation is coming soon.")


# -- notes ---------------------------------------------------------------------

_NOTES: dict[str, Note] = {}


def list_notes() -> list[Note]:
    return list(_NOTES.values())


def add_note(title: str, content: str) -> Note:
    note = Note(id=uuid.uuid4().hex[:8], title=title.strip() or "Untitled note", content=content)
    _NOTES[note.id] = note
    return note


def remove_note(note_id: str) -> bool:
    return _NOTES.pop(note_id, None) is not None
