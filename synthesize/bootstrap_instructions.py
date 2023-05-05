import os
import json
import random
import re
import string
from tqdm import tqdm
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[0]

HUMAN_QUERY_AUGMENTATIONS = str(ROOT_DIR / "data/seed-datasets/query_exemplars.jsonl")
HUMAN_RATIONALES = str(ROOT_DIR / "data/seed-datasets/rationale_exemplars.jsonl")
GPT3_QUERY_AUGMENTATIONS = str(ROOT_DIR / "data/synthetic-datasets/gpt3_query_exemplars.jsonl")
GPT3_RATIONALES = str(ROOT_DIR / "data/synthetic-datasets/gpt3_rationale_exemplars.jsonl")

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
    prompt += "please reason through the provided documentation to arrive at the answer, and print the answer at the end of the output:"
    for idx, (query, retrieval, rationale),  in enumerate(rationale_exemplars):
        query = re.sub(r"\s+", " ", query).strip().rstrip(":")
        retrieval = re.sub(r"\s+", " ", retrieval).strip().rstrip(":")
        rationale = re.sub(r"\s+", " ", rationale).strip().rstrip(":")
        prompt += f"Query: {query}\n"
        prompt += f"Retrieval: {retrieval}\n"
        prompt += f"Rationale: {rationale}\n"
    input_query = re.sub(r"\s+", " ", input_query).strip().rstrip(":")
    input_retrieval = re.sub(r"\s+", " ", input_retrieval).strip().rstrip(":")
    prompt += f"Query: {input_query}\n"
    prompt += f"Retrieval: {input_retrieval}\n"
    prompt += "Rationale: "
    return prompt


def synthesize_queries(input_queries, exemplars_per_prompt=6):
    args = parse_args()

    print("Synthesizing new queries...")
    print("Synthesizing new queries...", file=sys.stderr)

    seeds = [json.loads(l) for l in HUMAN_QUERY_AUGMENTATIONS]
    seed_exemplars = [(exemplar["query"], exemplar["augmentation"]) for exemplar in seeds]
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars")
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars", file=sys.stderr)

    gpt3_seed_exemplars = []
    with open(GPT3_QUERY_AUGMENTATIONS, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            gpt3_seed_exemplars.append((exemplar['query'], exemplar['augmentation']))
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars")
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars", file=sys.stderr)

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
        synthetic = []
        for input_query, result in zip(input_queries, results):
            augmentation = post_process_gpt3_response(result["response"])
            print(f"Writing new augmentation\n")
            print(f" || Original query: {input_query}\n")
            print(f" || New augmentation: {augmentation}\n\n\n")
            fout.write(json.dumps({
                "query": input_query
                "augmentation": augmentation
            }) + "\n")


def retrieve_for_query(input_queries):
    #TODO
    input_retrievals = None
    return input_retrievals


def synthesize_rationales(input_queries, exemplars_per_prompt=4):
    args = parse_args()

    print("Synthesizing new rationales...")
    print("Synthesizing new rationales...", file=sys.stderr)

    seeds = [json.loads(l) for l in HUMAN_RATIONALES]
    seed_exemplars = [(exemplar["query"], exemplar["augmentation"]) for exemplar in seeds]
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars")
    print(f"Loaded {len(seed_exemplars)} human-written seed exemplars", file=sys.stderr)

    gpt3_seed_exemplars = []
    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            gpt3_seed_exemplars.append((exemplar['query'], exemplar['augmentation']))
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars")
        print(f"Loaded {len(gpt3_seed_exemplars)} synthetic seed exemplars", file=sys.stderr)

    input_retrievals = retrieve_for_query(input_queries)

    with open(GPT3_RATIONALES, "a") as fout:
        batch_prompts = []
        for input_query, input_retrieval in zip(input_queries, input_retrievals):
            sample_synthetic = random.sample(gpt3_seed_exemplars, min(2, len(gpt3_seed_exemplars)))
            sample_human = random.sample(seed_exemplars, exemplars_per_prompt - len(sample_synthetic))
            rationale_exemplars = random.shuffle(sample_synthetic + sample_human)
            prompt = encode_queries(rationale_exemplars, input_query, input_retrieval)
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
        synthetic = []
        for input_query, input_retrieval, result in zip(input_queries, input_retrievals, results):
            rationale = post_process_gpt3_response(result["response"])
            print(f"Writing new rationale\n")
            print(f" || Original query: {input_query}\n")
            print(f" || Original retrieval: {input_retrieval}\n")
            print(f" || New rationale: {rationale}\n\n\n")
            fout.write(json.dumps({
                "query": input_query,
                "retrieval": input_retrieval,
                "rationale": rationale 
            }) + "\n")
    
    return


def post_process_gpt3_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_response = response["choices"][0]["text"]
    raw_response = re.sub(r"\s+", " ", raw_response).strip()
    return raw_response


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
    synthesize_queries()