import requests

API_URL = "https://api-inference.huggingface.co/models/barghavani/English_to_Hindi"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def translation(input_text):
    payload = {"inputs": input_text}
    response = query(payload)
    print(response)
    try:
        return response[0]['generated_text']
    except KeyError:
        return "Translation failed"