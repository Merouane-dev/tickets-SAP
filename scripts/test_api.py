# test_api.py
import requests

url = "http://127.0.0.1:5000/predict"

# Exemple de payload conforme à ton encodage (adapte selon tes colonnes)
data = {
    "TYPE": "Technical issue",
    "PRIORITE": "Critical",
    "JOUR_OUVERTURE": 2,
    "HEURE_OUVERTURE": 14
}

response = requests.post(url, json=data)
print("Status:", response.status_code)
print("Response:", response.json())
