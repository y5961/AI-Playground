import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT

# טעינת מפתח ה-API מקובץ .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def translate_to_cli(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input}
            ],
            temperature=0 # חשוב לדיוק מקסימלי במשימות טכניות
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# יצירת ממשק Gradio
demo = gr.Interface(
    fn=translate_to_cli,
    inputs=gr.Textbox(label="הוראה בשפה טבעית", placeholder="למשל: מה ה-IP שלי?"),
    outputs=gr.Code(label="פקודת CLI מוצעת"),
    title="Natural Language to CLI Agent",
    description="הקלידו הוראה בעברית וקבלו פקודה להרצה בטרמינל של Windows."
)

if __name__ == "__main__":
    demo.launch()