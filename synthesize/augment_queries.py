import openai
import json
import torch
from datasets import load_dataset

# Load the datasets
tldr = load_dataset("neulab/tldr")
conala = load_dataset("neulab/docprompting-conala") 

# Define the GPT-3.5 API call function
def call_gpt35_api(prompts, model="gpt-3.5-turbo"):
    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompts},
        ]   
    )

    #print(response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content'].strip().split("\n-")

# Define the processing function
def process_dataset(dataset, prompt_instruction):
    
    for split in dataset.keys():
        nl_col_responses = []

        print(len(dataset[split]))

        for i in range(0, len(dataset[split]), 20):
            batch = dataset[split].select(range(i, min(i+20, len(dataset[split]))))
            
            unique_rows = {row['nl']: idx for idx, row in enumerate(batch)}
            prompts = [f"\n-({nl}, " for nl in unique_rows.keys()]
            final_prompt = f"{prompt_instruction}" + "".join(prompts)
            
            responses = call_gpt35_api(final_prompt)
            while len(responses) != len(prompts):
                print('had to rereun')
                print(prompts, responses)
                responses = call_gpt35_api(final_prompt)
            
            print("equal length")
            response_dict = {nl: response for nl, response in zip(unique_rows.keys(), responses)}
            nl_col_responses.extend([response_dict[row['nl']] for row in batch])

        # Add a new column with GPT-3.5 responses
        dataset[split] = dataset[split].add_column("gpt35_response", nl_col_responses)

        # Save the dataset in .jsonl format
        dataset[split].to_json("output_{}.jsonl".format(split), orient="records", lines=True)

# Define the instructions for each dataset
tldr_instruction = "Rewrite this request to expand on the subfunctionalities required to implement this task:"
conala_instruction = ("We are performing enhancement on coding problem statements to make them clearer to students. Specifically we want to enhance each statement with the various steps and functionalities required for the problem by simply restating and expanding upon text pulled from the original statement. You should not mention any explicit function names, only what these functions should do. The post enhancement should be the pre enhancement plus some new text. Here are three (pre enhancement, post enhancement examples): "
 " (\"Create list `instancelist` containing 29 objects of type MyClass\", \"Create list `instancelist` containing 29 objects of type MyClass. This requires creating list ` instancelist `, creating 29 objects of type MyClass, and adding them to the list.\")"
 " (\"Taking the results of a bash command \"awk '{print $10, $11}' test.txt > test2.txt\"\", \"Taking the results of a bash command \"awk '{print $10, $11}' test.txt > test2.txt\". This requires executing bash command  \"awk '{print $10, $11}' test.txt > test2.txt\" and storing the results.\")"
 " (\"Save matplotlib graph to image file `filename.png` at a resolution of `300 dpi`\", \"Save matplotlib graph to image file `filename.png` at a resolution of `300 dpi`. This requires saving matplotlib graph to image file `filename.png` and specifying the resolution of `300 dpi`\")"
 "Now provide post-enhancements for the following in a bullet list: ")

# Process and save the datasets
# process_dataset(tldr, tldr_instruction)
process_dataset(conala, conala_instruction)