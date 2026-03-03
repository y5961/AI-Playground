"""
LlamaIndex script to index Markdown files with Custom Metadata.
Categorizes files by Agentic Tools (Cursor, Kiro, Claude) and uses local storage.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.cohere import CohereEmbedding

# טעינת משתני סביבה
load_dotenv()

# הגדרות קבועות
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STORAGE_DIR = "./storage"
DOCS_PATH = "./recipeProject/documentation"

def validate_env_variables():
    """וידוא שמפתח ה-API קיים."""
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in .env file")
    print("[OK] API keys loaded successfully")

def get_file_metadata(file_path):
    """
    פונקציית עזר להוספת Metadata לכל קובץ נטען.
    מזהה את הכלי (Tool) לפי שם הקובץ.
    """
    file_name = os.path.basename(file_path)
    metadata = {"file_name": file_name}
    
    # לוגיקה לזיהוי הכלי האוטומטי
    fname_lower = file_name.lower()
    if "cursor" in fname_lower:
        metadata["tool"] = "Cursor"
    elif "kiro" in fname_lower:
        metadata["tool"] = "Kiro"
    elif "claude" in fname_lower:
        metadata["tool"] = "Claude Code"
    else:
        metadata["tool"] = "General Documentation"
        
    return metadata

def initialize_embedding_model():
    """הגדרת מודל ה-Embeddings של Cohere."""
    return CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name="embed-multilingual-v3.0",
        input_type="search_document"
    )

def load_documents_with_metadata(path):
    """טעינת קבצי Markdown עם ה-Metadata המותאם."""
    print(f"Loading files from {path}...")
    reader = SimpleDirectoryReader(
        input_dir=path, 
        required_exts=[".md"],
        file_metadata=get_file_metadata # כאן מוזרק ה-Metadata
    )
    documents = reader.load_data()
    print(f"[OK] Loaded {len(documents)} document(s)")
    return documents

def create_and_persist_index(documents, embed_model):
    """יצירת האינדקס ושמירתו בתיקיית storage."""
    # יצירת הקשר אחסון (כברירת מחדל שומר לקבצי JSON)
    storage_context = StorageContext.from_defaults()
    
    # בניית האינדקס מהמסמכים
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    # שמירה פיזית בדיסק
    Path(STORAGE_DIR).mkdir(exist_ok=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[OK] Index stored locally in '{STORAGE_DIR}'")
    return index

def run_sample_queries(index):
    """הרצת שאילתות בדיקה שמשתמשות במידע החדש."""
    test_queries = [
        "What is the recipe structure according to the documentation?",
        "Which specific tools are mentioned in the metadata?",
        "How does the tagging system work?"
    ]
    
    print("\n" + "=" * 50)
    print("RUNNING TEST QUERIES")
    print("=" * 50)
    
    try:
        query_engine = index.as_query_engine(similarity_top_k=3)
        for q in test_queries:
            response = query_engine.query(q)
            print(f"\n❓ Query: {q}")
            print(f"💡 Response: {response}")
    except Exception as e:
        print(f"[WARN] Skipping sample query execution: {str(e)}")

def main():
    try:
        validate_env_variables()
        
        # 1. הכנת המודל
        embed_model = initialize_embedding_model()
        
        # 2. טעינת נתונים עם Metadata
        documents = load_documents_with_metadata(DOCS_PATH)
        
        if not documents:
            print("[ERROR] No markdown files found.")
            return

        # 3. אינדוקס ושמירה
        index = create_and_persist_index(documents, embed_model)
        
        # 4. בדיקת תוצאות
        run_sample_queries(index)
        
        print("\n[SUCCESS] Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()