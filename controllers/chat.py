import os
import requests


async def classifyQuery(query: str):
    headers = {
        "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
    }
    payload = {
        "inputs": query,
        "parameters": {
            "candidate_labels": [
                "A standalone general query",
                "A request to summarize content",
                "A follow-up query depending on previous context",
                "A request to generate an image",
            ]
        },
    }
    response = requests.post(
        os.environ["CLASSIFICATION_API_URL"], headers=headers, json=payload
    )
    return response.json()

