import os
import asyncio
import gradio as gr
from dotenv import load_dotenv

# ייבוא רכיבי LlamaIndex Workflow
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event
from llama_index.core import StorageContext, load_index_from_storage, Settings, PromptTemplate
from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from structured_knowledge import (
    load_structured_knowledge,
    should_use_structured_route,
    query_structured_knowledge,
)

load_dotenv(override=True)

# --- 1. ה-Prompt המקצועי המעודכן ---
QA_PROMPT_TMPL = (
    "אתה עוזר וירטואלי מקצועי עבור מערכת RecipeOrganizer.\n"
    "תפקידך לספק תשובות מדויקות על סמך קטעי המידע הבאים:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "הנחיות:\n"
    "1. אם המשתמש שואל על תהליכים (כמו 'איך עובד חיפוש'), הסבר את השלבים המופיעים בטקסט.\n"
    "2. ענה בעברית רהוטה ומקצועית.\n"
    "3. אם המידע לא קיים בטקסט, אמור במפורש שאינך יודע.\n\n"
    "השאלה: {query_str}\n\n"
    "תשובה:"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

# --- 2. הגדרת האירועים ---
class ValidationPassed(Event): query: str
class RoutedQuery(Event): query: str; route: str
class RetrievalDone(Event): query: str; context: str; nodes: list
class QualityPassed(Event): query: str; context: str; nodes: list

# --- 3. בניית ה-Workflow ---
class RecipeWorkflow(Workflow):
    def __init__(self, index, structured_data, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.structured_data = structured_data

    @step
    async def validate(self, ev: StartEvent) -> ValidationPassed | StopEvent:
        query = ev.get("query", "")
        if not query or len(query.strip()) < 2:
            return StopEvent(result="אנא הקלידי שאלה מפורטת יותר.")
        return ValidationPassed(query=query)

    @step
    async def route(self, ev: ValidationPassed) -> RoutedQuery:
        route = "structured" if should_use_structured_route(ev.query) else "semantic"
        return RoutedQuery(query=ev.query, route=route)

    @step
    async def retrieve(self, ev: RoutedQuery) -> RetrievalDone | StopEvent:
        if ev.route == "structured":
            structured_response = query_structured_knowledge(ev.query, self.structured_data)
            if structured_response:
                return StopEvent(result=structured_response)

        # שליפת המידע הכי רלוונטי
        nodes = self.index.as_retriever(similarity_top_k=1).retrieve(ev.query)
        if not nodes:
            return StopEvent(result="לא נמצא מידע רלוונטי במאגר.")
        context = "\n\n".join([n.get_content() for n in nodes])
        return RetrievalDone(query=ev.query, context=context, nodes=nodes)

    @step
    async def evaluate(self, ev: RetrievalDone) -> QualityPassed | StopEvent:
        # בדיקה שיש מספיק תוכן לענות
        if len(ev.context) < 40:
            return StopEvent(result="המידע שנמצא דל מדי כדי לספק תשובה איכותית.")
        return QualityPassed(query=ev.query, context=ev.context, nodes=ev.nodes)

    @step
    async def synthesize(self, ev: QualityPassed) -> StopEvent:
        from llama_index.core.base.llms.types import ChatMessage
        
        full_prompt = QA_PROMPT.format(context_str=ev.context, query_str=ev.query)
        messages = [ChatMessage(role="user", content=full_prompt)]
        response = await Settings.llm.achat(messages)
        
        # חילוץ שמות הקבצים (המקורות)
        sources_list = {n.metadata.get('file_name', 'קובץ ללא שם') for n in ev.nodes}
        sources_str = "\n".join([f"📄 {s}" for s in sources_list])
        
        final_answer = f"{str(response.message.content)}\n\n---\n**🔍 מקורות המידע שנבדקו:**\n{sources_str}"
        return StopEvent(result=final_answer)

# --- 4. אתחול המערכת ---
def init_system():
    cohere_key = os.getenv("COHERE_API_KEY")
    cohere_model = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")
    Settings.llm = Cohere(api_key=cohere_key, model=cohere_model)
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=cohere_key, 
        model_name="embed-multilingual-v3.0",
        input_type="search_query"
    )
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    structured_data = load_structured_knowledge()
    return RecipeWorkflow(index=index, structured_data=structured_data, timeout=60)

workflow_instance = init_system()

# --- 5. פונקציית הממשק ---
async def chat_handler(message, history):
    try:
        result = await workflow_instance.run(query=message)
        return str(result)
    except Exception as e:
        return f"שגיאה: {str(e)}"

# --- 6. הגדרת ממשק Gradio ללא דוגמאות ---
with gr.Blocks(title="Recipe Assistant") as demo:
    gr.Markdown("# 👨‍🍳 עוזר המערכת החכם")
    gr.Markdown("הקלידי שאלה למטה כדי לקבל תשובה מתוך תיעוד המערכת.")
    
    gr.ChatInterface(
        fn=chat_handler,
        # הסרנו את ה-examples כדי שלא יציע שאלות מראש
    )

if __name__ == "__main__":
    # כאן אנחנו מגדירים את ה-theme כדי לפתור את בעיית ה-init
    demo.launch(theme=gr.themes.Soft())