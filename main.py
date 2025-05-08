import requests

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
    result = query_ollama("Explain what is e-commerce")
    print(result)