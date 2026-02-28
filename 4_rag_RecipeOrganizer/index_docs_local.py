"""
LlamaIndex script to index Markdown files from ./documentation using Cohere embeddings.
Uses local vector storage (no Pinecone required) for easier testing and development.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.embeddings.cohere import CohereEmbedding

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
STORAGE_DIR = "./storage"

def validate_env_variables():
    """Validate that all required environment variables are set."""
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in .env file")
    print("[OK] API keys loaded successfully")


def initialize_embedding_model():
    """Initialize Cohere embedding model."""
    embedding_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_document"
    )
    print("[OK] Cohere embedding model initialized")
    return embedding_model


def load_markdown_files(docs_path: str = "./documentation"):
    """Load all Markdown files from the documentation directory."""
    docs_dir = Path(docs_path)
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documentation directory not found: {docs_path}")
    
    # Use SimpleDirectoryReader to load all markdown files
    reader = SimpleDirectoryReader(input_dir=str(docs_dir), required_exts=[".md"])
    documents = reader.load_data()
    
    print(f"[OK] Loaded {len(documents)} document(s) from {docs_path}")
    for doc in documents:
        print(f"     - {doc.metadata.get('file_name', 'Unknown')}")
    return documents


def create_vector_store_index(documents, embedding_model):
    """Create a VectorStoreIndex with local storage."""
    
    # Ensure storage directory exists
    Path(STORAGE_DIR).mkdir(exist_ok=True)
    
    # Create storage context with default vector store
    storage_context = StorageContext.from_defaults()
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True
    )
    
    # Persist the index locally
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"[OK] VectorStoreIndex created and stored in '{STORAGE_DIR}'")
    return index


def query_index(index, query_text: str, top_k: int = 3):
    """Query the index with a sample query."""
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query_text)
    
    print(f"\n[QUERY] {query_text}")
    print(f"[RESPONSE]\n{response}\n")
    return response


def main():
    """Main function to orchestrate the indexing pipeline."""
    try:
        print("=" * 70)
        print("LlamaIndex Cohere Local Vector Store Indexing")
        print("=" * 70 + "\n")
        
        # Validate environment setup
        validate_env_variables()
        
        # Initialize Cohere embeddings
        embedding_model = initialize_embedding_model()
        
        # Load documentation files
        documents = load_markdown_files("./documentation")
        
        if not documents:
            print("[ERROR] No markdown files found in ./documentation")
            return
        
        # Create and populate vector store index
        index = create_vector_store_index(documents, embedding_model)
        
        # Test the index with sample queries
        print("\n" + "=" * 70)
        print("Testing index with sample queries")
        print("=" * 70)
        query_index(index, "What is the recipe structure?")
        query_index(index, "How does the tagging system work?")
        query_index(index, "Explain the search logic")
        
        print("\n" + "=" * 70)
        print("[SUCCESS] Indexing complete!")
        print(f"Vector store persisted to: {STORAGE_DIR}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
