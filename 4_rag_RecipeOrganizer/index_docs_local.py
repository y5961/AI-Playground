"""
LlamaIndex script to index Markdown files with Custom Metadata.
Categorizes files by Agentic Tools (Cursor, Kiro, Claude) and uses local storage.
"""

import os
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding

# טעינת משתני סביבה
load_dotenv()

# הגדרות קבועות
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STORAGE_DIR = "./storage"
DOCS_PATH = "./recipeProject/documentation"
MIN_REQUIRED_TOOLS = 2

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
    normalized_path = str(file_path).replace("\\", "/")
    metadata = {
        "file_name": file_name,
        "source_path": normalized_path,
    }
    
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

def extract_title_from_text(text: str, file_name: str) -> str:
    """Extract title from first markdown heading, fallback to file stem."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                return title
    return Path(file_name).stem

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

    for doc in documents:
        file_name = doc.metadata.get("file_name", "untitled.md")
        doc.metadata["title"] = extract_title_from_text(doc.text or "", file_name)

    discovered_tools = {d.metadata.get("tool") for d in documents if d.metadata.get("tool")}
    if len(discovered_tools) < MIN_REQUIRED_TOOLS:
        raise ValueError(
            f"At least {MIN_REQUIRED_TOOLS} tools are required, but only found: {sorted(discovered_tools)}"
        )

    print(f"[OK] Loaded {len(documents)} document(s)")
    return documents


def print_tools_summary(documents):
    tool_counts = Counter(doc.metadata.get("tool", "Unknown") for doc in documents)
    print("\n" + "=" * 50)
    print("TOOLS DETECTION SUMMARY")
    print("=" * 50)
    for tool_name, count in sorted(tool_counts.items(), key=lambda item: item[0]):
        print(f"- {tool_name}: {count} document(s)")
    print(f"Total tools detected: {len(tool_counts)}")

def create_and_persist_index(documents, embed_model):
    """יצירת האינדקס ושמירתו בתיקיית storage."""
    # יצירת הקשר אחסון (כברירת מחדל שומר לקבצי JSON)
    storage_context = StorageContext.from_defaults()
    
    splitter = SentenceSplitter(chunk_size=700, chunk_overlap=80)

    # בניית האינדקס מהמסמכים
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
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

        print_tools_summary(documents)

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