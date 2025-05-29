import requests
import time

url = "http://localhost:5000/predict"
data = {"url": "https://example.com"}
headers = {"Content-Type": "application/json"}

for i in range(100):
    try:
        r = requests.post(url, json=data, headers=headers)
        print(f"{i+1}: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"{i+1}: Error - {e}")
    time.sleep(0.05)  # adjust or remove delay to increase pressure