import os
import sys
import json
import random
import re
import string
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pprint import pprint
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

HUMAN_QUERY_AUGMENTATIONS = str(ROOT_DIR / "synthesize/data/seed-datasets/query_exemplars.jsonl")
HUMAN_RATIONALES = str(ROOT_DIR / "synthesize/data/seed-datasets/rationale_exemplars.jsonl")
GPT3_QUERY_AUGMENTATIONS = str(ROOT_DIR / "synthesize/data/synthetic-datasets/gpt3_query_exemplars.jsonl")
GPT3_RATIONALES = str(ROOT_DIR / "synthesize/data/synthetic-datasets/gpt3_rationale_exemplars.jsonl")
STAR_RATIONALES = str(ROOT_DIR / "synthesize/data/synthetic-datasets/star_rationale_exemplars.jsonl")

TRAIN_ORACLE = str(ROOT_DIR / "docprompting/data/conala/cmd_train.oracle_man.full.json")
RETRIEVAL_RESULTS = str(ROOT_DIR / "docprompting/data/conala/retrieval_results.json")
CODE_DESCRIPTIONS = str(ROOT_DIR / "docprompting/data/conala/conala_docs.json")

FIRST_PARA_IDS = str(ROOT_DIR / "docprompting/data/conala-modified/python_manual_firstpara.tok.id")
FIRST_PARA_DESCRIPTIONS = str(ROOT_DIR / "docprompting/data/conala-modified/python_manual_firstpara.tok.txt")

TOP_K = 3

'''
Hard-coded prompts were based directly on prompts that Self-Instruct and CoT did
'''
def encode_queries(query_exemplars, input_query):
    # input format is:
    # query_exemplars should be a list of tuples of the form (query, augmentation)
    # query_exemplars can be any combination & length of human- or machine-written exemplars; this is decided exogenously to this function
    prompt = "You are an expert language model in code generation. "
    prompt += "Come up with clarifying restatements to queries for code generation problems, where these queries will be inputted into a retrieval algorithm for code documentation. "
    prompt += "Given a query for a coding task, please provide additional clarifications and details on the steps required to satisfy the query without mentioning explicit function names:"
    for idx, (query, augmentation) in enumerate(query_exemplars):
        query = re.sub(r"\s+", " ", query).strip().rstrip(":")
        augmentation = re.sub(r"\s+", " ", augmentation).strip().rstrip(":")
        prompt += f"Query: {query}\n"
        prompt += f"Augmentation: {augmentation}\n"
    input_query = re.sub(r"\s+", " ", input_query).strip().rstrip(":")
    prompt += f"Query: {input_query}"
    prompt += "Augmentation: "
    return prompt


def encode_rationales(rationale_exemplars, input_query, input_retrieval):
    # input format is:
    # rationale_exemplars should be a list of tuples of the form (query, retrieval, augmentation)
    # rationale_exemplars can be any combination & length of human- or machine-written exemplars; this is decided exogenously to this function
    # retrievals in rationale_exemplars should be genuine retrievals from a fully-trained SimCSE; please enforce this exogenously
    prompt = "You are an expert language model in code generation. "
    prompt += "Come up with a series of rationales for code generation problems under the following specification. "
    prompt += "Given a query for a coding task and a list of code documentation, "
    prompt += "please reason through the provided documentation to arrive at the answer code, "
    prompt += "and print the answer at the end of the output, in the format \"Therefore, the answer is \" with your answer code stated afterwards.\n\n"
    for idx, (query, retrieval, rationale) in enumerate(rationale_exemplars):
        query = re.sub(r"\s+", " ", query).strip().rstrip(":")
        retrieval = re.sub(r"\s+", " ", retrieval).strip().rstrip(":")
        rationale = re.sub(r"\s+", " ", rationale).strip().rstrip(":")
        prompt += f"Query: {query}\n"
        prompt += f"Relevant documentation: {retrieval}\n"
        prompt += f"Rationale: {rationale}\n\n"
    input_query = re.sub(r"\s+", " ", input_query).strip().rstrip(":")
    input_retrieval = re.sub(r"\s+", " ", input_retrieval).strip().rstrip(":")
    prompt += f"Query: {input_query}\n"
    prompt += f"Relevant code documentation: {input_retrieval}\n"
    prompt += "Rationale: "
    return prompt


def synthesize_queries(queries, exemplars_per_prompt=6):
    # input_queries should take the form of a list of tuples. [(query_id, query), ...]

    args = parse_args()

    print("Synthesizing new queries...")
    print("Synthesizing new queries...", file=sys.stderr)

    seed_exemplars = []
    with open(HUMAN_QUERY_AUGMENTATIONS, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            seed_exemplars.append((exemplar["query"], exemplar["augmentation"]))
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars")
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars", file=sys.stderr)

    gpt3_seed_exemplars = []
    with open(GPT3_QUERY_AUGMENTATIONS, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            gpt3_seed_exemplars.append((exemplar['query'], exemplar['augmentation']))
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars")
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars", file=sys.stderr)
    
    query_ids, input_queries = [t[0] for t in queries], [t[1] for t in queries]

    with open(GPT3_QUERY_AUGMENTATIONS, "a") as fout:
        batch_prompts = []
        for input_query in input_queries:
            sample_synthetic = random.sample(gpt3_seed_exemplars, min(2, len(gpt3_seed_exemplars)))
            sample_human = random.sample(seed_exemplars, exemplars_per_prompt - len(sample_synthetic))
            query_exemplars = random.shuffle(sample_synthetic + sample_human)
            prompt = encode_queries(query_exemplars, input_query)
            batch_prompts.append(prompt)
        results = make_gpt3_requests(
            engine=args.engine,
            prompts=batch_prompts,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["Query:", "Query :", "\n\n"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=args.api_key,
            organization=args.organization,
        )
        for query_id, input_query, result in zip(query_ids, input_queries, results):
            augmentation = post_process_gpt3_response(result["response"])
            print(f"Writing new augmentation\n")
            print(f" || Original query: {input_query}\n")
            print(f" || New augmentation: {augmentation}\n\n\n")
            fout.write(json.dumps({
                "question_id": query_id,
                "query": input_query,
                "augmentation": augmentation,
            }) + "\n")
    
    return


def retrieve_for_query(query_ids, truncate=False):
    with open(RETRIEVAL_RESULTS, "r") as fin:
        results_base = json.load(fin)
    functions_for_retrieval = [results_base[query_id]['retrieved'][:TOP_K] for query_id in query_ids]
    if truncate:
        with open(FIRST_PARA_IDS, "r") as fin:
            descriptions_functions = [line.rstrip('\n') for line in fin.readlines()]
        with open(FIRST_PARA_DESCRIPTIONS, "r") as fin:
            descriptions_descriptions = [line.rstrip('\n') for line in fin.readlines()]
        assert len(descriptions_functions) == len(descriptions_descriptions)
        descriptions_base = {f: d for f, d in zip(descriptions_functions, descriptions_descriptions)}
    else:
        with open(CODE_DESCRIPTIONS, "r") as fin:
            descriptions_base = json.load(fin)
    descriptions_for_retrieval = [[descriptions_base[f] for f in single_retrieval] for single_retrieval in functions_for_retrieval]
    retrievals = []
    for f_list, d_list in zip(functions_for_retrieval, descriptions_for_retrieval):
        current_retrieval = "\n"
        for f, d in zip(f_list, d_list):
            current_retrieval += f + "    "
            current_retrieval += d
        retrievals.append(current_retrieval)
    return retrievals


def synthesize_rationales(queries, exemplars_per_prompt=4):
    # input_queries should take the form of a list of tuples. [(query_id, query), ...]

    args = parse_args()

    print("Synthesizing new rationales...")
    print("Synthesizing new rationales...", file=sys.stderr)

    seed_exemplars = []
    with open(HUMAN_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            seed_exemplars.append((exemplar["query"], exemplar["retrieval"], exemplar["rationale"]))
        print(f"Loaded {len(seed_exemplars)} human-written seed exemplars")
        print(f"Loaded {len(seed_exemplars)} human-written seed exemplars", file=sys.stderr)

    gpt3_seed_exemplars = []
    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            gpt3_seed_exemplars.append((exemplar['query'], exemplar['retrieval'], exemplar['rationale']))
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars")
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars", file=sys.stderr)

    query_ids, input_queries = [t[0] for t in queries], [t[1] for t in queries]
    input_retrievals = retrieve_for_query(query_ids)

    print(len(input_queries))
    print(len(input_retrievals))

    with open(GPT3_RATIONALES, "a") as fout:
        batch_prompts = []
        for input_query, input_retrieval in zip(input_queries, input_retrievals):            
            sample_synthetic = random.sample(gpt3_seed_exemplars, min(2, len(gpt3_seed_exemplars)))
            sample_human = random.sample(seed_exemplars, exemplars_per_prompt - len(sample_synthetic))
            rationale_exemplars = sample_synthetic + sample_human
            random.shuffle(rationale_exemplars)
            prompt = encode_rationales(rationale_exemplars, input_query, input_retrieval)
            return prompt
            batch_prompts.append(prompt)
        return
        results = make_gpt3_requests(
            engine=args.engine,
            prompts=batch_prompts,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["Query:", "Query :", "\n\n"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=args.api_key,
            organization=args.organization,
        )
        for query_id, input_query, input_retrieval, result in zip(query_ids, input_queries, input_retrievals, results):
            rationale = post_process_gpt3_response(result["response"])
            print(f"Writing new rationale\n")
            print(f" || Original query: {input_query}\n")
            print(f" || Original retrieval: {input_retrieval}\n")
            print(f" || New rationale: {rationale}\n\n\n")
            fout.write(json.dumps({
                "question_id": query_id,
                "query": input_query,
                "retrieval": input_retrieval,
                "rationale": rationale,
            }) + "\n")
    
    return


def star_for_code(exemplars_per_prompt=4):
    args = parse_args()

    print("Patching incorrect synthetic samples with rationalization...")
    print("Patching incorrect synthetic samples with rationalization...", file=sys.stderr)

    human_rationales = []
    with open(HUMAN_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            human_rationales.append((exemplar["query"], exemplar["retrieval"], exemplar["rationale"]))
        print(f"Loaded {len(human_rationales)} human-written seed exemplars")
        print(f"Loaded {len(human_rationales)} human-written seed exemplars", file=sys.stderr)

    gpt3_rationales = []
    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            gpt3_rationales.append((exemplar["question_id"], exemplar['query'], exemplar['retrieval'], exemplar['rationale']))
        print(f"Loaded {len(gpt3_rationales)} synthetic rationales")
        print(f"Loaded {len(gpt3_rationales)} synthetic rationales", file=sys.stderr)
    
    answer_key = {}
    with open(TRAIN_ORACLE, "r") as fin:
        answers = json.load(fin)
        for sample in answers:
            answer_key[sample['question_id']] = sample['cmd']

    with open(STAR_RATIONALES, "w") as fout:
        star_rationales = ""
        to_be_rationalized = []
        for query_id, query, retrieval, rationale in gpt3_rationales:
            rationale = re.sub(r"\s+", " ", rationale).strip().rstrip('.').rstrip(':')
            needs_rationalization = not rationale.endswith(answer_key[query_id])
            if needs_rationalization:
                to_be_rationalized += (query_id, query, retrieval)
            else:
                star_rationales += json.dumps({
                    "question_id": query_id,
                    "query": query,
                    "retrieval": retrieval,
                    "rationale": rationale,
                }) + "\n"
        
        if not to_be_rationalized:
            fout.write(star_rationales)
            return

        batch_prompts = []
        for _, query, retrieval in to_be_rationalized:
            plus_hint = query + f"(Hint: the answer is {answer_key[query_id]})\n"
            
            sample_synthetic = random.sample(gpt3_rationales, min(2, len(gpt3_rationales)))
            sample_human = random.sample(human_rationales, exemplars_per_prompt - len(sample_synthetic))
            rationale_exemplars = random.shuffle(sample_synthetic + sample_human)
            prompt = encode_rationales(rationale_exemplars, plus_hint, retrieval)
            batch_prompts.append(prompt)

        results = make_gpt3_requests(
            engine=args.engine,
            prompts=batch_prompts,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=["Query:", "Query :", "\n\n"],
            logprobs=1,
            n=1,
            best_of=1,
            api_key=args.api_key,
            organization=args.organization,
        )

        for (query_id, query, retrieval), result in zip(to_be_rationalized, results):
            rationalization = post_process_gpt3_response(result["response"])
            rationalization = re.sub(r"\s+", " ", rationalization).strip().rstrip('.').rstrip(':')
            maybe_correct = rationalization.endswith(answer_key[query_id])
            if maybe_correct:
                star_rationales += json.dumps({
                    "question_id": query_id,
                    "query": query,
                    "retrieval": retrieval,
                    "rationale": rationalization,
                }) + "\n"
                print(f"Query {query_id} successfully rationalized")
                print(f"Query {query_id} successfully rationalized", file=sys.stderr)
            else:
                print(f"Rationale from query {query_id} still wrong after rationalization; cutting from dataset...")
                print(f"Rationale from query {query_id} still wrong after rationalization; cutting from dataset...", file=sys.stderr)
        
        fout.write(star_rationales)
    
    return


def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_response = response["choices"][0]["text"]
    raw_response = re.sub(r"\s+", " ", raw_response).strip()
    return raw_response


def script_synthesize_conala_train():
    finished_query_ids = set()

    with open(HUMAN_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            finished_query_ids.add(exemplar["question_id"])

    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            finished_query_ids.add(exemplar["question_id"])
    
    all_query_ids = set()
    query_id_to_query = {}
    with open(TRAIN_ORACLE, "r") as fin:
        full_oracle = json.load(fin)
        for exemplar in full_oracle:
            all_query_ids.add(exemplar["question_id"])
            query_id_to_query[exemplar["question_id"]] = exemplar["nl"]
    all_query_ids = list(all_query_ids - finished_query_ids)
    all_queries = [query_id_to_query[id] for id in all_query_ids]

    return synthesize_rationales(list(zip(all_query_ids, all_queries)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt-3.5-turbo",
        help="The engine to use."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not provided, synthetic data generation will not run."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not provided, synthetic data generation will not run."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # print(script_synthesize_conala_train())
    pprint(retrieve_for_query(["348196-52"], truncate=True))