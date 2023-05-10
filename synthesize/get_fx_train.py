import openai
import json
import torch
from datasets import load_dataset

conala = load_dataset("neulab/docprompting-conala") 
train = conala['train']

fxn_set = set()
with open(f'data/seen_fxn.txt', 'w') as f:
    for row in train:
        for fxn in row['oracle_man']:
            if fxn not in fxn_set:
                fxn_set.add(fxn)
                f.write(fxn + '\n') 

    