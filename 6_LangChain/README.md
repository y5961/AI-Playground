# 📚 NotebookLM - LangChain RAG Application

אפליקציה חכמה לחיפוש וניתוח מסמכים עם AI, בשלוש שכבות עיצור מתגדלות.

---

## 🎯 מטרת הפרויקט

**NotebookLM** הוא עוזר מחקר חכם שמסוגל:

1. **💬 Chat עם זיכרון** - שיחה מובנית עם הקשר (thread-based history)
2. **📖 RAG (Retrieval-Augmented Generation)** - חיפוש סמנטי במקורות המעוברות
3. **🌐 Web Search** - חיפוש בנושאים בצורה עמוקה מהרשת כולה

המערכת משלבת:
- **LangGraph** - אורכסטרציה של סוכנים חכמים
- **OpenAI** - LLM ו-embeddings
- **FastAPI** - REST API לשימוש מקצועי
- **Firecrawl** - web scraping וחילוץ תוכן

---

## 🏗️ ארכיטקטורה

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│  POST /api/chat  |  GET /api/sources  |  /api/sources/  │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴────────────┐
        │                       │
    ┌───▼─────────┐      ┌─────▼──────────┐
    │  Chat Agent │      │  Source Store  │
    │  (LangGraph)│      │  (VectorStore) │
    └───┬─────────┘      └─────┬──────────┘
        │                      │
        │                  ┌───┴──────────┬────────────┐
        │                  │              │            │
    ┌───▼──────┐      ┌────▼────┐   ┌────▼──┐    ┌───▼───┐
    │  OpenAI  │      │  Upload │   │Search │    │  Web  │
    │   LLM    │      │  Files  │   │Local  │    │Search │
    │ gpt-4o-m │      │         │   │Source │    │(FC)   │
    └──────────┘      └─────────┘   └───────┘    └───────┘
```

---

## ✨ תכונות ראשיות

### Step 1: 💬 Chat Agent (בסיס)
- שיחה זלולה עם ה-LLM
- זיכרון שיחה מבוסס `thread_id`
- System prompt בעברית

### Step 2: 📖 RAG (חיפוש מקומי)
- העלאת קבצים (טקסט, PDF, וכו')
- חיפוש סמנטי חכם
- טיפול אוטומטי בחלוקה (chunking)
- ציטוט מדויק של מקורות

### Step 3: 🌐 Web Search (Firecrawl)
- חיפוש עמוק בנושאים
- מספר שאילתות להכסוה בזוויות שונות
- Web scraping וחילוץ תוכן
- אינדקס אוטומטי ל-VectorStore
- שילוב תוצאות רשת עם מקורות מקומיים

---

## 🚀 התחלה מהירה

### דרישות

- **Python 3.13+**
- **UV** (package manager)
- **API Keys:**
  - `OPENAI_API_KEY` - מ-[OpenAI](https://platform.openai.com)
  - `FIRECRAWL_API_KEY` - מ-[Firecrawl](https://app.firecrawl.dev) (אופציונלי)

### 1️⃣ התקנה

```bash
# שכפל את הפרויקט
cd "c:\Users\User\Documents\Handesaim\שנה ב\מלכה ברוק\6_LangChain"

# ודא שה-dependencies מותקנים
uv sync

# הגדר את ה-.env עם ה-API keys שלך
# (העתק את המפתח מ-OpenAI ל-.env)
```

### 2️⃣ הפעלת הסרוור

```bash
# בטרמינל אחד - הפעל את FastAPI
uv run uvicorn src.api.serve:app --reload --port 8000
```

הסרוור יהיה זמין ב: **http://localhost:8000**

### 3️⃣ בדיקה מהירה

```bash
# בטרמינל אחר - הפעל את הבדיקות
uv run python src/app.py

# או בדיקות web search בלבד
uv run python src/test_web_search.py
```

---

## 📖 שימוש

### דרך REST API

#### 1. הוספת מקור (קובץ)

```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LangChain Guide",
    "content": "LangChain is a framework for developing applications..."
  }'

# Response:
# {
#   "id": "src_abc123",
#   "name": "LangChain Guide",
#   "active": true
# }
```

#### 2. חיפוש בנושא מהרשת (Web Search)

```bash
curl -X POST http://localhost:8000/api/sources/search-web \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "LangChain",
    "num_sources": 3
  }'

# Response:
# {
#   "topic": "LangChain",
#   "sources_added": 3,
#   "source_ids": ["web_1", "web_2", "web_3"],
#   "status": "success"
# }
```

#### 3. שאלה לסוכן (Chat)

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "מה זה LangChain?",
    "thread_id": "user_123"
  }'

# Response:
# {
#   "answer": "LangChain הוא framework...",
#   "sources": [
#     {
#       "id": "src_abc123",
#       "name": "LangChain Guide",
#       "excerpt": "LangChain is a framework..."
#     }
#   ]
# }
```

#### 4. קבלת רשימת מקורות

```bash
curl http://localhost:8000/api/sources

# Response:
# [
#   {
#     "id": "src_abc123",
#     "name": "LangChain Guide",
#     "active": true,
#     "char_count": 2500
#   },
#   ...
# ]
```

### דרך Python Script

```python
from api.services import add_source, search_web, run_chat
from api.schemas import WebSearchRequest, ChatRequest

# הוספת מקור
source_id = add_source("My Document", "Content here...")

# חיפוש בנושא מהרשת
result = search_web(WebSearchRequest(topic="Python", num_sources=3))
print(f"נוספו {result.sources_added} מקורות")

# שאלה לסוכן
response = run_chat(ChatRequest(
    question="מה זה Python?",
    thread_id="user_123"
))
print(response.answer)
print(response.sources)
```

---

## 📁 מבנה הפרויקט

```
6_LangChain/
├── README.md                    # ← הדוקומנטציה
├── pyproject.toml              # Python dependencies
├── .env                        # API keys (שימור מקומי)
│
├── src/
│   ├── __init__.py
│   ├── app.py                  # בדיקות ראשוניות
│   ├── test_web_search.py      # בדיקות web search
│   │
│   ├── agents/                 # Chat agent
│   │   ├── __init__.py
│   │   └── chat.py             # LangGraph agent עם tools
│   │
│   ├── core/                   # Core functionality
│   │   ├── __init__.py
│   │   ├── sources.py          # Chunking & formatting
│   │   ├── store.py            # VectorStore management
│   │   ├── web_search.py       # Firecrawl integration
│   │   └── web_sources.py      # Web source manager
│   │
│   └── api/                    # FastAPI REST API
│       ├── __init__.py
│       ├── app.py              # FastAPI application
│       ├── serve.py            # ASGI server entry
│       ├── schemas.py          # Pydantic models
│       └── services.py         # Business logic
│
├── client/                     # (עתידי) Frontend
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
└── main.py                     # Entry point
```

---

## 🔧 הגדרות מתקדמות

### הגדרת .env

```env
# OpenAI (חובה)
OPENAI_API_KEY=sk-proj-...

# Firecrawl (עבור web search)
FIRECRAWL_API_KEY=fc-...

# אופציונלי - מקדימות קודמות
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...
```

### שינוי המודל

ב-`src/agents/chat.py`:
```python
# שנה את ה-model
llm = ChatOpenAI(
    model="gpt-4",  # או gpt-4o, gpt-3.5-turbo
    temperature=0.7
)
```

### שינוי גודל ה-chunks

ב-`src/core/sources.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # שנה לגודל רצוי
    chunk_overlap=200
)
```

---

## 🧪 בדיקות

### בדיקה 1: Chat בסיסי
```bash
uv run python src/app.py
```

### בדיקה 2: Web Search
```bash
uv run python src/test_web_search.py
```

### בדיקה 3: API locally
```bash
# הפעל את הסרוור
uv run uvicorn src.api.serve:app --reload

# בחלון נפרד, בדיקה
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"שלום","thread_id":"test"}'
```

---

## 🌐 Firecrawl Setup (Web Search)

### שלב 1: יצירת חשבון
1. עבור ל-[app.firecrawl.dev](https://app.firecrawl.dev)
2. יצור חשבון חדש
3. ודא שה-Tools מופעלים: ✅ Search, ✅ Scrape, ✅ Crawl

### שלב 2: העתקת ה-API Key
1. בדף ההגדרות, העתק את ה-API key
2. הדבק ב-`.env`:
   ```env
   FIRECRAWL_API_KEY=your-actual-key-here
   ```

### שלב 3: בדיקה
```bash
uv run python src/test_web_search.py
```

---

## 🔍 איך זה עובד

### זרימת Chat עם RAG:

```
[User Question]
      ↓
[Search Local Sources] ← חיפוש סמנטי
      ↓
[Found Documents?]
      ├─ YES → [Add to Context]
      └─ NO  → [General Knowledge]
      ↓
[Generate Answer] ← gpt-4o-mini
      ↓
[Return with Citations] ← ממקורות וקישורים
```

### זרימת Web Search:

```
[Topic from User]
      ↓
[Generate 5 Search Queries] ← covering different angles
      ↓
[Search Each Query] ← via Firecrawl
      ↓
[Filter Quality Results]
      ↓
[Scrape Top-3] ← extract markdown
      ↓
[Chunk & Embed] ← OpenAI embeddings
      ↓
[Index to VectorStore]
      ↓
[Agent can now answer from Web]
```

---

## 📊 דוגמאות שימוש

### דוגמה 1: ניתוח מסמך

```python
from api.services import add_source, run_chat
from api.schemas import ChatRequest

# הוסף מסמך
doc_id = add_source("Python Guide", """
Python is a high-level programming language...
It's known for its simplicity and readability...
""")

# שאל על המסמך
response = run_chat(ChatRequest(
    question="מה יתרונות Python?",
    thread_id="session_1"
))

print(response.answer)
# → "לפי המסמך, Python מוכר בפשטותו ובקריאות שלו..."
```

### דוגמה 2: מחקר אונליין

```python
from api.services import search_web, run_chat
from api.schemas import WebSearchRequest, ChatRequest

# חפש בנושא
search_web(WebSearchRequest(
    topic="Machine Learning Applications",
    num_sources=5
))

# שאל בעניין
response = run_chat(ChatRequest(
    question="מה הן היישומים העיקריים של Machine Learning?",
    thread_id="session_1"
))

print(response.answer)
# → "בהתבסס על החיפוש שביצעתי בנושא:
#    - Computer Vision: ...
#    - Natural Language Processing: ...
#    - ..."
```

### דוגמה 3: שיחה מ-thread ארוכה

```python
from api.services import run_chat
from api.schemas import ChatRequest

thread = "researcher_123"

# שאלה 1
r1 = run_chat(ChatRequest(
    question="מה זה AI?",
    thread_id=thread
))

# שאלה 2 - יחזיק בהקשר מ-שאלה 1
r2 = run_chat(ChatRequest(
    question="וכיצד זה קשור ל-Machine Learning?",  # זוכר את השיחה הקודמת
    thread_id=thread
))
```

---

## 🐛 פתרון בעיות

### בעיה: "ModuleNotFoundError: No module named 'core'"
**פתרון:** ודא שאתה בתיקייה הנכונה
```bash
cd "c:\Users\User\Documents\Handesaim\שנה ב\מלכה ברוק\6_LangChain"
uv run python src/app.py
```

### בעיה: "Firecrawl API key not configured"
**פתרון:** עדכן את `.env`:
```env
FIRECRAWL_API_KEY=your-actual-key-from-firecrawl
```

### בעיה: OpenAI API errors
**פתרון:** בדוק את ה-API key:
```bash
echo $env:OPENAI_API_KEY  # Windows PowerShell
echo $OPENAI_API_KEY      # Linux/Mac
```

### בעיה: VectorStore ריק
**פתרון:** הוסף מקורות תחילה:
```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","content":"Hello world"}'
```

---

## 🎓 למד עוד

- **LangChain:** [docs.langchain.com](https://docs.langchain.com)
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **Firecrawl:** [docs.firecrawl.dev](https://docs.firecrawl.dev)
- **FastAPI:** [fastapi.tiangolo.com](https://fastapi.tiangolo.com)

---

## 📝 License

פרויקט אישי לתרגול. שימוש חופשי.

---

## 👨‍💻 פיתוח

**Step 1:** ✅ Chat Agent  
**Step 2:** ✅ RAG + Vector Search  
**Step 3:** ✅ Web Search (Firecrawl)  
**Step 4:** 🔜 Structured Artifacts (Summaries, FAQs, etc.)  
**Step 5:** 🔜 Frontend (React/Vue)  

---

**קוד ברור = מחקר טוב! 🧠**
