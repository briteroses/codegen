import requests
import json

params = {
    'dataset': 'neulab/docprompting-conala',
}

response = requests.get('https://datasets-server.huggingface.co/splits', params=params)
json
