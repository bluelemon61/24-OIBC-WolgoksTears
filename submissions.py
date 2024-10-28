import json 
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

result = {
    'submit_result' : [0.0] * 24 
}

success = requests.post(f'{API_URL}/submissions/cmpt-2024',
                    data=json.dumps(result),
                    headers={
                        'Authorization': f'Bearer {API_KEY}'
                    }).json()

print(success) 
