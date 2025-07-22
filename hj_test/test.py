import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "Qwen/Qwen2_5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "보험금이란?"}
        ],
        "temperature": 0.7,
        "stream": False
    }
)

print(response.json())