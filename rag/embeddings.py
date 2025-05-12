from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "chroma_db"

def index_documents(chunks, persist_directory=CHROMA_PATH):
    """Indexes document chunks into the Chroma vector store."""
    print(f"Indexing {len(chunks)} chunks...")
    # Use from_documents for initial creation.
    # This will overwrite existing data if the directory exists but isn't a valid Chroma DB.
    # For incremental updates, initialize Chroma first and use vectorstore.add_documents().
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_function(),
        persist_directory=persist_directory,
    )
    print(f"Indexing complete. Data saved to: {persist_directory}")
    return vectorstore


def get_embedding_function(model_name="nomic-embed-text"):
    """Initializes the Ollama embedding function."""
    embeddings = OllamaEmbeddings(model=model_name)
    print(f"Initialized Ollama embeddings with model: {model_name}")
    return embeddings

def get_vector_store(persist_directory=CHROMA_PATH):
    """Initializes or loads the Chroma vector store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embedding_function(),
    )
    print(f"Vector store initialized/loaded from: {persist_directory}")
    return vectorstore