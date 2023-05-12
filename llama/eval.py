"""
- wrangle bleu-4, recall, recall_unseen into some nice modular functions
input: json file with one dictionary
    the keys are metadata on the experiment, like i can do a tuple "(n icl exemplars, 
    true/false retrieval, true/false rationale)"
    the values are a list of 3-tuples, each 3-tuple is a pairing (question id, LLM output code) 
    snippet as a string, oracle/ground truth code snippet as a string), 
    and the list spans the whole eval set for that experiment
output: 3-tuple of float values, (bleu_4, recall, recall_unseen)
"""

### READ IN SEEN/UNSEEN SPLIT
file_name = 'data/seen_fxn.txt'
with open(file_name) as file:
    SEEN_FXN = [line.rstrip() for line in file]

import json
from collections import OrderedDict
import evaluate

import numpy as np

TOP_K = [1, 3, 5, 8, 10, 12, 15, 20, 30, 50, 100, 200]

def calc_recall(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_hit = sum([x in cur_result_vids for x in oracle_man])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / (len(oracle_man)) if len(oracle_man) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}

def calc_recall_unseen(src, pred, print_result=True, top_k=None):
    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        ###FILTER TO UNSEEN ONLY
        oracle_man_unseen = [x for x in oracle_man if x not in SEEN_FXN]

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_result_vids_unseen = [x for x in cur_result_vids if x not in SEEN_FXN]
            cur_hit = sum([x in cur_result_vids_unseen for x in oracle_man_unseen])
            # recall_n[tk] += cur_hit / (len(oracle_man) + 1e-10)
            recall_n[tk] += cur_hit / (len(oracle_man_unseen)) if len(oracle_man_unseen) else 1
            precision_n[tk] += cur_hit / tk
    recall_n = {k: v / len(pred) for k, v in recall_n.items()}
    precision_n = {k: v / len(pred) for k, v in precision_n.items()}

    if print_result:
        for k in sorted(recall_n.keys()):
            print(f"{recall_n[k] :.3f}", end="\t")
        print()
        for k in sorted(precision_n.keys()):
            print(f"{precision_n[k] :.3f}", end="\t")
        print()
        for k in sorted(recall_n.keys()):
            print(f"{2 * precision_n[k] * recall_n[k] / (precision_n[k] + recall_n[k] + 1e-10) :.3f}", end="\t")
        print()

    return {'recall': recall_n, 'precision': precision_n}

def eval_retrieval_from_file(data_file, retrieval_file, question_ids,
                             oracle_entry='oracle_man', retrieval_entry='retrieved', top_k=None):

    assert 'oracle_man.full' in data_file or 'conala' not in data_file, (data_file)
    # for conala
    with open(data_file, "r") as f:
        d = json.load(f)

    ### TODO: calculate gold such that it only includes question_ids in question_ids
    gold = [item[oracle_entry] for item in d]

    with open(retrieval_file, "r") as f:
        r_d = json.load(f)
    pred = [r_d[q_id][retrieval_entry] for q_id in question_ids]

    recall_metrics = calc_recall(gold, pred, top_k=top_k, print_result=False)
    recall_metrics_unseen = calc_recall_unseen(gold, pred, top_k=top_k, print_result=False)
    return recall_metrics, recall_metrics_unseen

'''
retrieval_file: dictionary of question_id: retrieved documentation
'''
def get_metrics(results_file, retrieval_file):
    ### LOAD IN JSON
    with open(results_file, 'r') as file:
        # Read the contents of the file
        json_string = file.read()

    # Parse the JSON string into a dictionary
    data = json.loads(json_string)

    # Create a new dictionary variable and assign the parsed data
    experiments = dict(data)

    results_dict = {}

    for experiment in experiment:
        dataset = experiments[experiment]
        predictions = []
        references = []
        question_ids = []
        ### loop thru dataset, get predictions and references
        for line in dataset:
            question_ids.append(line[0])
            predictions.append(line[1])
            references.append(line[2])

        bleu = evaluate.load("neulab/python_bleu")
        results = bleu.compute(predictions=predictions, references=references)
        bleu_4 = results["bleu_score"]

        recall_metrics, recall_metrics_unseen = eval_retrieval_from_file('data/conala/cmd_test.oracle_man.full', retrieval_file, question_ids)

        recall = recall_metrics['recall'][1]
        recall_unseen = recall_metrics_unseen['recall'][1]

        results_dict[experiment] = (bleu_4, recall, recall_unseen)

    return results_dict