"""
LlamaIndex script to index Markdown files from ./documentation to Pinecone using Cohere embeddings.
Reads all .md files, generates embeddings via Cohere, and stores them in Pinecone vector store.
API keys are loaded from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from pinecone import Pinecone
import ssl
import urllib3

# Disable SSL warnings and certificate verification for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "recipe-organizer")

def validate_env_variables():
    """Validate that all required environment variables are set."""
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY not found in .env file")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    print("✓ API keys loaded successfully")


def initialize_embedding_model():
    """Initialize Cohere embedding model."""
    embedding_model = CohereEmbedding(
        cohere_api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
        input_type="search_document"
    )
    print("✓ Cohere embedding model initialized")
    return embedding_model


def initialize_pinecone():
    """Initialize Pinecone client and connect to index."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"✓ Pinecone client initialized")
        
        # Try to get the index directly
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✓ Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
        return index
    except Exception as e:
        print(f"❌ Pinecone connection error: {str(e)}")
        print("Please ensure:")
        print("1. PINECONE_API_KEY is valid in .env")
        print(f"2. Index '{PINECONE_INDEX_NAME}' exists in your Pinecone project")
        print("3. You have stable internet connection")
        raise


def load_markdown_files(docs_path: str = "./documentation"):
    """Load all Markdown files from the documentation directory."""
    docs_dir = Path(docs_path)
    
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documentation directory not found: {docs_path}")
    
    # Use SimpleDirectoryReader to load all markdown files
    reader = SimpleDirectoryReader(input_dir=str(docs_dir), required_exts=[".md"])
    documents = reader.load_data()
    
    print(f"✓ Loaded {len(documents)} document(s) from {docs_path}")
    return documents


def create_vector_store_index(documents, embedding_model, pinecone_index):
    """Create a VectorStoreIndex and store documents in Pinecone."""
    
    # Initialize Pinecone vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        add_sparse_vector=False
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True
    )
    
    print(f"✓ VectorStoreIndex created and stored in Pinecone")
    return index


def query_index(index, query_text: str, top_k: int = 3):
    """Query the index with a sample query."""
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(query_text)
    
    print(f"\n📋 Query: {query_text}")
    print(f"📌 Response:\n{response}\n")
    return response


def main():
    """Main function to orchestrate the indexing pipeline."""
    try:
        # Validate environment setup
        validate_env_variables()
        
        # Initialize Cohere embeddings
        embedding_model = initialize_embedding_model()
        
        # Initialize Pinecone
        pinecone_index = initialize_pinecone()
        
        # Load documentation files
        documents = load_markdown_files("./documentation")
        
        # Create and populate vector store index
        index = create_vector_store_index(documents, embedding_model, pinecone_index)
        
        # Test the index with a sample query
        print("\n" + "="*60)
        print("Testing index with sample queries...")
        print("="*60)
        query_index(index, "What is the recipe structure?")
        query_index(index, "How does the tagging system work?")
        
        print("\n✅ Indexing complete! Your documentation is now searchable in Pinecone.")
        
    except Exception as e:
        print(f"❌ Error during indexing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
