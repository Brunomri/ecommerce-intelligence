import requests

from data_processing.embeddings import get_embedding_function, index_documents
from data_processing.load_data import load_data


def query_ollama(prompt, model="qwen3:8b"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    data_chunks = load_data()
    vector_store = index_documents(data_chunks, get_embedding_function())
    result = query_ollama("Explain what is e-commerce")
    print(result)