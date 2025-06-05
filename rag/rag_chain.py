from datetime import datetime
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama


def query_rag(question, vector_store, result_type, result_path):
    """Queries the RAG chain and prints the response."""
    rag_chain = create_rag_chain(vector_store)
    print("\nQuerying RAG chain...")
    print(f"\nQuestion: {question}")
    response = rag_chain.invoke(question)
    print(f"\nResponse: {response}")

    current_time = datetime.now().isoformat().split("T")
    file_name = f"{result_type}_{current_time[0]}_{current_time[1]}.txt"

    path = Path(result_path)
    path.mkdir(parents=True, exist_ok=True)

    with open(f"{result_path}{file_name}", "w") as result_file:
        result_file.write(f"Question:\n{question}\nResponse:\n{response}")


def create_rag_chain(vector_store):
    """Creates the RAG chain."""
    rag_chain = ({
        "context": create_retriever(vector_store),
        "question": RunnablePassthrough(),
    } | create_prompt_template() | initialize_llm() | StrOutputParser())
    print("Rag chain created")
    return rag_chain

def initialize_llm(model_name="qwen3:8b", temperature=0, context_window=8192):
    """Initialize Ollama with the LLM provided."""
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=context_window,
    )
    print(f"Initialized ChatOllama with model: {model_name}, temperature: {temperature}, context window: {context_window}")
    return llm

# TODO: Review the need of top_chunks parameter
def create_retriever(vector_store, search_type="similarity", top_chunks=20):
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={'k':top_chunks}
    )
    print(f"Created retriever with search_type: {search_type}, retrieving the top {top_chunks} relevant chunks")
    return retriever

def create_prompt_template():
    template = """Answer the question based ONLY on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    print("Prompt template created")
    return prompt