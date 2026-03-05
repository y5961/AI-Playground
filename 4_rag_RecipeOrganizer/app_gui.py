import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage, get_response_synthesizer, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere 
from structured_knowledge import (
    load_structured_knowledge,
    should_use_structured_route,
    query_structured_knowledge,
)

load_dotenv(override=True)

# ה-Prompt המקצועי
QA_PROMPT_TMPL = (
    "אתה עוזר וירטואלי מקצועי עבור מערכת RecipeOrganizer.\n"
    "תפקידך לספק תשובות מדויקות ורהוטות על סמך קטעי המידע הבאים:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "הנחיות לעבודה:\n"
    "1. ענה תמיד ובאופן בלעדי בשפה שבה המשתמש שאל (אם פנו בעברית, ענה בעברית רהוטה).\n"
    "2. אל תעתיק טקסט גולמי בצורה מגושמת; נסח את המידע מחדש בצורה מקצועית ונקייה משגיאות.\n"
    "3. אם המידע לא מופיע בטקסט, ציין זאת.\n\n"
    "השאלה: {query_str}\n\n"
    "תשובה מקצועית:"
)
QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

def init_rag():
    cohere_key = os.getenv("COHERE_API_KEY")
    cohere_model = os.getenv("COHERE_CHAT_MODEL", "command-a-03-2025")
    
    # הגדרת מודלים
    Settings.embed_model = CohereEmbedding(
        cohere_api_key=cohere_key,
        model_name="embed-multilingual-v3.0",
        input_type="search_query"
    )
    # הגדרת ה-LLM עם הגדרות בסיסיות למניעת שגיאות גרסה
    Settings.llm = Cohere(api_key=cohere_key, model=cohere_model)
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
    return index

index = init_rag()
structured_data = load_structured_knowledge()

def ask_recipe_bot(question):
    try:
        if should_use_structured_route(question):
            structured_response = query_structured_knowledge(question, structured_data)
            if structured_response:
                return structured_response

        # בניית מנוע השאילתה בתוך ה-Try
        retriever = VectorIndexRetriever(index=index, similarity_top_k=1)
        synth = get_response_synthesizer(response_mode="compact", text_qa_template=QA_PROMPT)
        processor = SimilarityPostprocessor(similarity_cutoff=0.2)
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synth,
            node_postprocessors=[processor]
        )

        source_nodes = retriever.retrieve(question)
        source_nodes = processor.postprocess_nodes(source_nodes)

        if not source_nodes:
            return "לא נמצא מידע רלוונטי בתיעוד. נסה לשאול שאלה אחרת."

        try:
            response = query_engine.query(question)
        except Exception as llm_error:
            err_txt = str(llm_error)
            if "model" in err_txt.lower() and ("removed" in err_txt.lower() or "404" in err_txt):
                snippets = []
                for n in source_nodes[:2]:
                    snippet = n.node.get_content().strip()[:350]
                    snippets.append(snippet)
                sources = list(set([n.node.metadata.get('file_name', 'Unknown') for n in source_nodes]))
                return f"מודל הניסוח לא זמין כרגע, אבל נמצא מידע רלוונטי:\n\n" + "\n\n".join(snippets) + f"\n\n📌 **מקורות:** {', '.join(sources)}"
            raise
        
        if not response or not str(response).strip():
             return "המודל לא הצליח לגבש תשובה. נסי לשאול שוב או לבדוק את החיבור."

        sources = list(set([n.node.metadata.get('file_name', 'Unknown') for n in response.source_nodes]))
        return f"{str(response)}\n\n📌 **מקורות:** {', '.join(sources)}"

    except Exception as e:
        # הדפסה לדיבאג בטרמינל
        print(f"Detailed Error: {e}")
        return f"מצטער, חלה שגיאה טכנית בניסוח התשובה. (Error: {str(e)[:100]})"

# ממשק Gradio
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 עוזר המתכונים החכם שלך")
    input_box = gr.Textbox(label="מה תרצי לבדוק?")
    output_box = gr.Textbox(label="תשובה מהתיעוד:", lines=10)
    btn = gr.Button("שאל", variant="primary")
    btn.click(fn=ask_recipe_bot, inputs=input_box, outputs=output_box)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())