import json 
import openai 
import ast
import math
import concurrent.futures

tldr_json = json.load(open('data/tldr/manual_all_raw.json'))

sample_keys = [i for i in tldr_json.keys()][0:100]
        
# test openai call on first sample
openai.api_key = "sk______________________________________K"

def get_gpt_reponse(text, optional_start = None):
    if optional_start is None: 
        messages = [
            {"role": "system", "content": "Given the passage of documentation, extract a concise list of tuples containing 3 things: relevant information/documentation, description of the desired code, code/command. So, each tuple should have 3 strings, taking the format:  ('relevant information/documentation', 'description of desired code', ' code/command'). \n\nFor example the first tuple in the list of a hypothetical passage of pip (python package manager) could be: \n[(\"The python package manager 'pip' allows users to install packages with 'pip install <package-name>'. The '--user' flag installs packages locally for the user. \", \"install a package locally for the user\", \"pip install --user '{{package}}'\"),]\n\nRemember output format is ('relevant information/documentation', 'description of desired code', 'code/command') and every tuple should contain the code/command."},
            {"role": "user", "content": text},
        ]
    else: 
        messages = [
            {"role": "system", "content": "Given the passage of documentation, extract a concise list of tuples containing 3 things: relevant information/documentation, description of the desired code, code/command. So, each tuple should have 3 strings, taking the format:  ('relevant information/documentation', 'description of desired code', ' code/command'). \n\nFor example the first tuple in the list of a hypothetical passage of pip (python package manager) could be: \n[(\"The python package manager 'pip' allows users to install packages with 'pip install <package-name>'. The '--user' flag installs packages locally for the user. \", \"install a package locally for the user\", \"pip install --user '{{package}}'\"),]\n\nRemember output format is ('relevant information/documentation', 'description of desired code', 'code/command') and every tuple should contain the code/command."},
            {"role": "user", "content": text},
            {"role": "assistant", "content": optional_start}
        ]
        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.1,
        max_tokens=1400,
        top_p=1.0,
        frequency_penalty=0.1,
        presence_penalty=0.0
    )
    return response

def process_sample(key, total_text):
    response_storage = []
    initial_text = None
    num_splits = math.ceil(len(total_text)/7000)
    length_split = len(total_text)//num_splits
    split_text = [total_text[i:i+length_split] for i in range(0, len(total_text), length_split)]
    for i in range(num_splits):
        response = get_gpt_reponse(split_text[i], initial_text)
        if initial_text is not None:
            for option in range(len(response["choices"])): 
                response["choices"][option]["message"]["content"] = "[" + response["choices"][option]["message"]["content"]
        response_storage.append(response)
        try:
            if i == 0:
                initial_text = "[" + str(ast.literal_eval(response["choices"][0]["message"]["content"])[0]) + ", "
        except:
            pass
    return key, response_storage

# test, get response for all samples
sample_data = [(key, tldr_json[key]) for key in sample_keys]
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(process_sample, key, total_text) for key, total_text in sample_data]
    for future in concurrent.futures.as_completed(futures):
        try:
            key, response_storage = future.result()
            with open(f'custom_data/responses/{key}.json', 'w') as outfile:
                json.dump(response_storage, outfile)
        except Exception as e:
            print(f"!!!!!!!!!!!!!!Error on {key}!!!!!!!!!!!!!!")
            print(e)