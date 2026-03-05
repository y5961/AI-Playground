import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

DOCS_PATH = Path("./recipeProject/documentation")
STRUCTURED_STORE_PATH = Path("./storage/structured_knowledge.json")
SCHEMA_PATH = Path("./storage/structured_schema.json")

SCHEMA = {
    "schema_version": "1.0.0",
    "item_types": {
        "decisions": {
            "description": "Project decisions and design choices",
            "required_fields": ["id", "description", "source", "discovered_at"],
        },
        "rules": {
            "description": "Guidelines, requirements, constraints, and rules",
            "required_fields": ["id", "description", "source", "discovered_at"],
        },
        "warnings": {
            "description": "Warnings, limitations, errors, and caveats",
            "required_fields": ["id", "description", "source", "discovered_at"],
        },
    },
}

_DECISION_PATTERNS = [
    r"\bdecid(?:e|ed|ion)\b",
    r"\bdesign choice\b",
    r"\barchitecture\b",
    r"החלט",
]

_RULE_PATTERNS = [
    r"\bmust\b",
    r"\bshould\b",
    r"\brequired\b",
    r"\brule\b",
    r"\bguideline\b",
    r"\bconstraint\b",
    r"כלל",
    r"הנח",
    r"חובה",
]

_WARNING_PATTERNS = [
    r"\bwarning\b",
    r"\bcaution\b",
    r"\berror\b",
    r"\blimit(?:ation)?\b",
    r"\bdeprecated\b",
    r"שגיא",
    r"אזהר",
    r"מוגבל",
]


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -\t\n\r")


def _matches_any(text: str, patterns: List[str]) -> bool:
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _classify_line(line: str) -> str | None:
    if _matches_any(line, _WARNING_PATTERNS):
        return "warnings"
    if _matches_any(line, _RULE_PATTERNS):
        return "rules"
    if _matches_any(line, _DECISION_PATTERNS):
        return "decisions"
    return None


def _build_item(item_type: str, description: str, file_name: str, line_number: int) -> Dict:
    discovered_at = datetime.now(timezone.utc).isoformat()
    item_id = f"{item_type}-{Path(file_name).stem}-{line_number}"
    return {
        "id": item_id,
        "description": description,
        "source": {
            "file": file_name,
            "row": line_number,
        },
        "discovered_at": discovered_at,
    }


def _extract_from_document(file_name: str, text: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    decisions: List[Dict] = []
    rules: List[Dict] = []
    warnings: List[Dict] = []

    for idx, raw_line in enumerate(text.splitlines(), start=1):
        line = _normalize_line(raw_line)
        if len(line) < 15:
            continue

        item_type = _classify_line(line)
        if not item_type:
            continue

        item = _build_item(item_type, line, file_name, idx)
        if item_type == "decisions":
            decisions.append(item)
        elif item_type == "rules":
            rules.append(item)
        else:
            warnings.append(item)

    return decisions, rules, warnings


def build_structured_knowledge(
    docs_path: Path = DOCS_PATH,
    output_path: Path = STRUCTURED_STORE_PATH,
    schema_path: Path = SCHEMA_PATH,
) -> Dict:
    if not docs_path.exists():
        raise FileNotFoundError(f"Documentation path not found: {docs_path}")

    documents = sorted(docs_path.glob("*.md"))
    extracted = {
        "decisions": [],
        "rules": [],
        "warnings": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    for doc_path in documents:
        text = doc_path.read_text(encoding="utf-8")
        decisions, rules, warnings = _extract_from_document(doc_path.name, text)
        extracted["decisions"].extend(decisions)
        extracted["rules"].extend(rules)
        extracted["warnings"].extend(warnings)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(SCHEMA, ensure_ascii=False, indent=2), encoding="utf-8")
    return extracted


def load_structured_knowledge(store_path: Path = STRUCTURED_STORE_PATH) -> Dict:
    if not store_path.exists():
        return build_structured_knowledge()
    return json.loads(store_path.read_text(encoding="utf-8"))


def should_use_structured_route(question: str) -> bool:
    q = (question or "").lower()
    router_signals = [
        "latest",
        "newest",
        "last",
        "guideline",
        "rule",
        "rules",
        "warning",
        "warnings",
        "decision",
        "decisions",
        "list",
        "show all",
        "מה ההנחיה",
        "הנחיה",
        "אזהר",
        "כללים",
        "רשימה",
        "אילו",
    ]
    return any(token in q for token in router_signals)


def query_structured_knowledge(question: str, data: Dict) -> str | None:
    q = (question or "").lower()
    decisions = data.get("decisions", [])
    rules = data.get("rules", [])
    warnings = data.get("warnings", [])

    if "warning" in q or "אזהר" in q:
        target = warnings
        title = "Warnings"
    elif "decision" in q or "החלט" in q:
        target = decisions
        title = "Decisions"
    elif "rule" in q or "guideline" in q or "כלל" in q or "הנח" in q:
        target = rules
        title = "Rules"
    elif "list" in q or "רשימה" in q or "אילו" in q:
        target = rules + warnings + decisions
        title = "Structured Items"
    else:
        return None

    if not target:
        return "לא נמצאו פריטים מובנים מתאימים במסמכים."

    if "latest" in q or "newest" in q or "last" in q or "האחרו" in q or "עדכני" in q:
        latest_item = max(target, key=lambda item: item.get("discovered_at", ""))
        src = latest_item.get("source", {})
        return (
            f"הפריט העדכני ביותר ב-{title}:\n"
            f"- {latest_item.get('description', '')}\n"
            f"📄 מקור: {src.get('file', 'unknown')}:{src.get('row', '?')}"
        )

    top_items = target[:8]
    lines = [f"{title} שנמצאו:"]
    for item in top_items:
        src = item.get("source", {})
        lines.append(f"- {item.get('description', '')} ({src.get('file', 'unknown')}:{src.get('row', '?')})")
    return "\n".join(lines)
