import os
import sys
import json
import random
import re
import torch
import numpy as np
import transformers

import argparse
from pathlib import Path
from pprint import pprint
import glob

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

HUMAN_RATIONALES = str(ROOT_DIR / "synthesize/data/seed-datasets/rationale_exemplars.jsonl")
GPT3_RATIONALES = str(ROOT_DIR / "synthesize/data/synthetic-datasets/gpt3_rationale_exemplars.jsonl")
STAR_RATIONALES = str(ROOT_DIR / "synthesize/data/synthetic-datasets/star_rationale_exemplars.jsonl")

FOR_FINETUNING_AT = str(ROOT_DIR / "synthesize/data/for_finetuning.jsonl")

def prepare_for_finetuning():
    assistant = "You are an expert language model in code generation. "
    assistant += f"Come up with a rationale for a code generation problem under the following specification. "
    assistant += "Given a query for a coding task and a list of code documentation, "
    assistant += "please reason through the provided documentation to arrive at the answer code and "
    assistant += "print the answer at the end of the output. "
    assistant += "The final sentence in your response should state \"The answer is \" followed by the correct code snippet.\n\n"

    train_rationales = {}
    with open(HUMAN_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            train_rationales[exemplar["question_id"]] = (exemplar["query"], exemplar["retrieval"], exemplar["rationale"])
    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            train_rationales[exemplar["question_id"]] = (exemplar["query"], exemplar["retrieval"], exemplar["rationale"])
    
    for_finetuning = ""
    for icl_id, (icl_query, icl_retrieval, icl_rationale) in train_rationales.items():
        prompt = assistant
        prompt += f"Query: {icl_query}\n"
        prompt += f"Relevant code documentation: {icl_retrieval}\n"
        prompt += f"Rationale: "
        response = icl_rationale
        for_finetuning += json.dumps({
            'prompt': prompt,
            'response': response
        }) + "\n"
        pprint({
            'prompt': prompt,
            'response': response
        })
        return
    
    with open(FOR_FINETUNING_AT, "w") as fout:
        fout.write(for_finetuning)

    return

if __name__ == "__main__":
    prepare_for_finetuning()