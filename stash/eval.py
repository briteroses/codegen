import json
from collections import OrderedDict
import evaluate

import numpy as np
from pathlib import Path

"""
one eval pipeline for bleu-4, recall, recall_unseen.
input: json file with one dictionary
    the keys are metadata on the experiment, as a tuple (n icl exemplars, true/false retrieval, true/false rationale)
    the values are a list of 3-tuples, each 3-tuple is a pairing (question id, LLM output code snippet as a string, oracle/ground truth code snippet as a string), 
        and the list spans the whole eval set for that experiment
output: 3-tuple of float values, (bleu_4, recall, recall_unseen)
"""

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

TEST_ORACLE = str(ROOT_DIR / "docprompting/data/conala/cmd_test.oracle_man.full.json")
RETRIEVAL_RESULTS = str(ROOT_DIR / "docprompting/data/conala/retrieval_results.json")

### READ IN SEEN/UNSEEN SPLIT
SEEN_FXN_AT = str(ROOT_DIR / 'llama/data/seen_fxn.txt')

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

    with open(SEEN_FXN_AT) as fin:
        seen_fxn = [line.rstrip() for line in fin]

    top_k = TOP_K if top_k is None else top_k
    recall_n = {x: 0 for x in top_k}
    precision_n = {x: 0 for x in top_k}

    for s, p in zip(src, pred):
        # cmd_name = s['cmd_name']
        oracle_man = s
        pred_man = p

        ###FILTER TO UNSEEN ONLY
        oracle_man_unseen = [x for x in oracle_man if x not in seen_fxn]

        for tk in recall_n.keys():
            cur_result_vids = pred_man[:tk]
            cur_result_vids_unseen = [x for x in cur_result_vids if x not in seen_fxn]
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


def pass_at_k(k=10):


def exact_match(results_file):
    


def eval_retrieval_from_file(question_ids, top_k=None):

    assert 'oracle_man.full' in data_file or 'conala' not in data_file, (data_file)
    # for conala
    with open(TEST_ORACLE, "r") as f:
        d = json.load(f)

    ### TODO: calculate gold such that it only includes question_ids in question_ids
    gold = [item['oracle_man'] for item in d if item['question_id'] in question_ids]

    with open(RETRIEVAL_RESULTS, "r") as f:
        r_d = json.load(f)
    pred = [r_d[q_id]['retrieved'] for q_id in question_ids]

    recall_metrics = calc_recall(gold, pred, top_k=top_k, print_result=False)
    recall_metrics_unseen = calc_recall_unseen(gold, pred, top_k=top_k, print_result=False)
    return recall_metrics, recall_metrics_unseen


def get_metrics(results_file):
    ### LOAD IN JSON
    with open(results_file, 'r') as fin:
        # Read the contents of the file
        json_string = fin.read()

    # Parse the JSON string into a dictionary
    experiments = json.load(json_string)

    results_dict = {}

    for experiment, results in experiments.items():
        predictions = []
        references = []
        question_ids = []
        ### loop thru dataset, get predictions and references
        for line in results:
            question_ids.append(line[0])
            predictions.append(line[1])
            references.append(line[2])

        bleu = evaluate.load("neulab/python_bleu")
        results = bleu.compute(predictions=predictions, references=references)
        bleu_4 = results["bleu_score"]

        recall_metrics, recall_metrics_unseen = eval_retrieval_from_file(question_ids)

        recall = recall_metrics['recall'][1]
        recall_unseen = recall_metrics_unseen['recall'][1]

        results_dict[experiment] = (bleu_4, recall, recall_unseen)

    return results_dict


