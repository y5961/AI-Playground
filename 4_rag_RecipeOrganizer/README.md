# app_workflow — RecipeOrganizer Agent

מסמך זה מתייחס **רק** לפרויקט `app_workflow.py`.

## מטרת הפרויקט

`app_workflow.py` מממש Agent מבוסס Workflow (RAG) שמקבל שאלה, שולף מידע רלוונטי מהתיעוד המקומי, בודק את איכות המידע, ומחזיר תשובה מנוסחת בעברית עם מקורות.

זרימת העבודה:
1. `validate` — אימות שהשאלה תקינה
2. `retrieve` — שליפה מהאינדקס המקומי (`storage/`)
3. `evaluate` — בדיקת איכות/מספיקות הקונטקסט
4. `synthesize` — ניסוח תשובה עם Cohere

## דרישות מקדימות

- Python 3.10+
- סביבת `venv` פעילה
- קובץ `.env` עם:

```env
COHERE_API_KEY=your_cohere_api_key
# אופציונלי:
COHERE_CHAT_MODEL=command-a-03-2025
```

## התקנה

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r recipeProject\requirements.txt
```

## אינדוקס התיעוד (לפני הרצה ראשונה)

```powershell
.\venv\Scripts\python.exe index_docs_local.py
```

## הרצת app_workflow

```powershell
.\venv\Scripts\python.exe app_workflow.py
```

או אם ה-venv כבר פעיל:

```powershell
python app_workflow.py
```

## דוגמאות לשאלות שה-Agent יודע לענות

- "איך עובד חיפוש לפי שם?"
- "מה מבנה המתכון לפי התיעוד?"
- "איך עובד מנגנון התגיות?"
- "אילו שדות חובה יש במתכון?"
- "מה נשמר עבור כל רכיב במתכון?"

## תקלות נפוצות

### `ModuleNotFoundError: No module named 'llama_index'`
הסיבה: הרצה עם Python מחוץ ל-venv.

פתרון:
```powershell
.\venv\Scripts\python.exe app_workflow.py
```

### שגיאת Cohere על מודל שהוסר (`404 model removed`)
הסיבה: שימוש במודל ישן (למשל `command-r`).

פתרון:
- לעבוד עם מודל נתמך (למשל `command-a-03-2025`)
- או להגדיר `COHERE_CHAT_MODEL` ב-`.env`

### "לא נמצא מידע רלוונטי"
- לוודא שהרצת אינדוקס: `index_docs_local.py`
- לוודא שהשאלה קשורה למסמכים ב-`recipeProject/documentation/`

## קבצים רלוונטיים ל-app_workflow

- `app_workflow.py`
- `index_docs_local.py`
- `storage/`
- `recipeProject/documentation/`
