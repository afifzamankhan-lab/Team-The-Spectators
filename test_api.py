import json
import requests

# 1. Load the sample cases provided by the organizers
with open('SUST_Preli_Sample_Cases.json', 'r', encoding='utf-8') as file:
    cases = json.load(file)

# 2. Grab the 'input' from the first sample case
first_test_case = cases[0]['input']

# 3. Send it to your local server
url = "http://localhost:8000/analyze-ticket"
response = requests.post(url, json=first_test_case)

# 4. Print the result
print(f"Status Code: {response.status_code}")
print(json.dumps(response.json(), indent=2))