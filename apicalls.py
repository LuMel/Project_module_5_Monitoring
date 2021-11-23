import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1"
PORT = ":8000"

with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
response1 = requests.get(URL + PORT + '/prediction').content
response2 = requests.get(URL + PORT + '/scoring').content
response3 = requests.get(URL + PORT + '/summarystats').content
response4 = requests.get(URL + PORT + '/diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open(model_path + '/apireturns.txt', 'wb') as f:
    for line in responses:
        f.write(line)
        f.write(b'\n')
