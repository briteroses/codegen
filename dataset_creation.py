import openai
import json
import torch
from datasets import load_dataset
from openai_secret_manager import secrets

# Load the datasets
tldr = load_dataset("neulab/tldr")
conala = load_dataset("neulab/docprompting-conala")

# Define the GPT-3.5 API call function
def call_gpt35_api(prompt, model="gpt-3.5-turbo"):
    openai.api_key = secrets["openai"]["api_key"]
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

# Define the processing function
def process_dataset(dataset, prompt_instruction):
    for split in dataset.keys():
        nl_col_responses = []

        for row in dataset[split]:
            prompt = f"{prompt_instruction} {row['nl']}"
            response = call_gpt35_api(prompt)
            nl_col_responses.append(response)

        # Add a new column with GPT-3.5 responses
        dataset[split] = dataset[split].add_column("gpt35_response", nl_col_responses)

        # Save the dataset in .jsonl format
        dataset[split].to_json("output_{}.jsonl".format(split), orient="records", lines=True)

# Define the instructions for each dataset
tldr_instruction = "Rewrite this request to expand on the subfunctionalities required to implement this task:"
conala_instruction = "We are performing enhancement on coding problem statements to make them clearer to students. Specifically we want to enhance each statement with the various steps and functionalities required for the problem by simply restating and expanding upon text pulled from the original statement. You should not mention any explicit function names, only what these functions should do. The post enhancement should be the pre enhancement plus some new text. Here are three (pre enhancement, post enhancement examples): " 
+ " (\"Create list `instancelist` containing 29 objects of type MyClass\", \"Create list `instancelist` containing 29 objects of type MyClass. This requires creating list ` instancelist `, creating 29 objects of type MyClass, and adding them to the list.\")" 
+ "(\"Taking the results of a bash command \"awk '{print $10, $11}' test.txt > test2.txt\"\", \"Taking the results of a bash command \"awk '{print $10, $11}' test.txt > test2.txt\". This requires executing bash command  \"awk '{print $10, $11}' test.txt > test2.txt\" and storing the results.\")"
+ "(\"Save matplotlib graph to image file `filename.png` at a resolution of `300 dpi`\", \"Save matplotlib graph to image file `filename.png` at a resolution of `300 dpi`. This requires saving matplotlib graph to image file `filename.png` and specifying the resolution of `300 dpi`\")"
+ "Now perform enhancements on the following: ("

# Process and save the datasets
# process_dataset(tldr, tldr_instruction)
process_dataset(conala, conala_instruction)