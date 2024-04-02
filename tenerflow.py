import requests

API_URL = "https://api-inference.huggingface.co/models/barghavani/English_to_Hindi"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

query({"inputs": "The answer to the universe is"})