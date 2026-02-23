import os
import ssl
import json
import requests
import re  # ספרייה לניקוי טקסט
from todo_service import add_task, get_tasks
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

API_URL = "https://router.huggingface.co/v1/chat/completions"


def agent(query):
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

    prompt = f"""
    You are a task manager.
    If the user wants to add a task, return: {{"action": "add", "task": "task name", "reply": "הוספתי את המשימה"}}
    If they want to see tasks, return: {{"action": "get", "reply": "הנה הרשימה שלך"}}
    User: "{query}"
    Return ONLY JSON.
    """

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.1  # טמפרטורה נמוכה הופכת את ה-AI ליותר מדויק
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        # חילוץ ה-JSON בעזרת Regex (מוצא את מה שבין הסוגריים המסולסלים)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())

            if data.get("action") == "add":
                task_title = data.get("task", "משימה חדשה")
                add_task(task_title)
                return data.get("reply", f"הוספתי: {task_title}")

            elif data.get("action") == "get":
                tasks = get_tasks()
                titles = [t['title'] for t in tasks]
                return f"{data.get('reply')}: " + (", ".join(titles) if titles else "הרשימה ריקה")

        return content  # אם זה לא JSON, לפחות נראה את הטקסט

    except Exception as e:
        return f"שגיאה בפענוח: {str(e)}"