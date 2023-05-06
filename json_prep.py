# write a script that reads in a json and prints the first 10 results

import json
import sys
from datasets import load_dataset

if len(sys.argv) != 2:
    print("Usage: python json_prep.py <json_file>")



#file_name = 'data/conala/python_manual_firstpara.tok.txt'# 
#file_name = 'data/conala/dev_retriever.json'
file_name = 'data/conala/conala_nl.txt'
encodings_to_try = [ 'ascii']
good_encodings = ['utf-8', 'is-8859-1', 'iso-8859-2', 'iso-8859-15', 'windows-1250', 'windows-1251', 'windows-1252']
content = None

for encoding in encodings_to_try:
    try:
        with open(file_name, 'r', encoding=encoding) as file:
            content = file.read()
        print(f"File successfully read using {encoding} encoding.")
        break
    except UnicodeDecodeError:
        print(f"Failed to read the file using {encoding} encoding.")
        continue

if content is not None:
    print(content)
else:
    print("Could not read the file using any of the tried encodings.")
#dataset = load_dataset('json', data_files=file_name)

print(len(content))
print(type(content))

# with open(file_name) as f:
#     dataset = []
#     for line in f:
#         print(line)
#         dataset.append(json.loads(line.decode('utf-8').strip()))
#     print(len(dataset))
#     #logger.info(f'size of the eval data: {len(dataset)}')

with open(file_name, 'r', encoding='ascii') as f:
    data = f.read()
    #content = f.read()
    print(data)
    print(data[:100])

print('here')
print(data.count('x'))

for i in range(10):
    print(content[i]['nl'])
    print(content[i]['gpt35_response'])
    print()