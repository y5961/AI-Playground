"""
LlamaIndex script to index Markdown files from ./documentation to Pinecone using Cohere embeddings.
Reads all .md files, generates embeddings via Cohere, and stores them in Pinecone vector store.
API keys are loaded from .env file.
"""

import os
import re
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
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
DOCS_PATH = "./recipeProject/documentation"
MIN_REQUIRED_TOOLS = 2

def get_file_metadata(file_path: str) -> dict:
    file_name = os.path.basename(file_path)
    normalized_path = str(file_path).replace("\\", "/")
    metadata = {
        "file_name": file_name,
        "source_path": normalized_path,
    }

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
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                return title
    return Path(file_name).stem

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
        model_name="embed-multilingual-v3.0",
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
    reader = SimpleDirectoryReader(
        input_dir=str(docs_dir),
        required_exts=[".md"],
        file_metadata=get_file_metadata,
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
    
    print(f"✓ Loaded {len(documents)} document(s) from {docs_path}")
    return documents


def print_tools_summary(documents):
    tool_counts = Counter(doc.metadata.get("tool", "Unknown") for doc in documents)
    print("\n" + "=" * 60)
    print("TOOLS DETECTION SUMMARY")
    print("=" * 60)
    for tool_name, count in sorted(tool_counts.items(), key=lambda item: item[0]):
        print(f"- {tool_name}: {count} document(s)")
    print(f"Total tools detected: {len(tool_counts)}")


def create_vector_store_index(documents, embedding_model, pinecone_index):
    """Create a VectorStoreIndex and store documents in Pinecone."""
    
    # Initialize Pinecone vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        add_sparse_vector=False
    )
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    splitter = SentenceSplitter(chunk_size=700, chunk_overlap=80)

    # Create index from documents
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model,
        transformations=[splitter],
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
        documents = load_markdown_files(DOCS_PATH)
        print_tools_summary(documents)
        
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
