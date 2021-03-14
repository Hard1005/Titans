import requests
from ibm_watson_machine_learning import APIClient
import pandas as pd
import json

df = pd.read_csv("Test Data.csv")
df = df.drop("Unnamed: 0" , axis=1)
df = df.drop("Total Exp" , axis=1)

to_score = df.values.tolist()

wml_creds = {
    "apikey": "AgOvNjeKYkIS_GqSXSl1TXxiGEAm875Q3wiB_i7LM5WH",
    "url": "https://jp-tok.ml.cloud.ibm.com"

}

wml_client = APIClient(wml_creds)
wml_client.spaces.list()

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "AgOvNjeKYkIS_GqSXSl1TXxiGEAm875Q3wiB_i7LM5WH"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"fields": list(df.columns), "values": to_score}]}

response_scoring = requests.post('https://jp-tok.ml.cloud.ibm.com/ml/v4/deployments/539516a4-77ea-4707-a330-b9a8d8e213fa/predictions?version=2021-03-13', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())