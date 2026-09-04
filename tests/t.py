import os
import requests
from dotenv import load_dotenv

# تحميل المتغيرات من ملف .env
load_dotenv()

# جلب مفتاح API
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY is missing in your .env file!")
else:
    print("Connecting to Groq servers to fetch available models...\n")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
    
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("✅ SUCCESS! Here are the exact model IDs you can use right now:")
        print("-" * 50)
        for model in models:
            print(f"- {model['id']}")
        print("-" * 50)
        print("Copy ONE of the names above and paste it into your generator.py file.")
    else:
        print(f"Failed to fetch models. Status Code: {response.status_code}")
        print(response.text)