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

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.load import load_model

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

LLAMA_AT = str(ROOT_DIR / "llama/llama_hf_7b")
ALPACA_7B_AT = str(ROOT_DIR / "llama/alpaca")
MOSAIC_INSTRUCT_AT = 'mosaicml/mpt-7b-instruct'
MOSAIC_CHAT_AT = 'mosaicml/mpt-7b-chat'
ALPACA_13B_AT = None

ALL_MODEL_NAMES = [LLAMA_AT, ALPACA_7B_AT, MOSAIC_INSTRUCT_AT, MOSAIC_CHAT_AT, ALPACA_13B_AT]

TEST_ORACLE = str(ROOT_DIR / "docprompting/data/conala/cmd_test.oracle_man.full.json")
HUMAN_RATIONALES = str(ROOT_DIR / "synthesize/data/seed-datasets/rationale_exemplars.jsonl")
GPT3_RATIONALES = str(ROOT_DIR / "synthesize/data/synthetic-datasets/gpt3_rationale_exemplars.jsonl")

RETRIEVAL_RESULTS = str(ROOT_DIR / "docprompting/data/conala/retrieval_results.json")
CODE_DESCRIPTIONS = str(ROOT_DIR / "docprompting/data/conala/conala_docs.json")
FIRST_PARA_IDS = str(ROOT_DIR / "docprompting/data/conala-modified/python_manual_firstpara.tok.id")
FIRST_PARA_DESCRIPTIONS = str(ROOT_DIR / "docprompting/data/conala-modified/python_manual_firstpara.tok.txt")
TOP_K = 3

'''
Taken from HuggingFace user hatimbr's final comment in
https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/6
'''
class StopOnString(StoppingCriteria):
    def __init__(self, stop_string, tokenizer, encounters=1):
        super().__init__()
        self.stop_token_seq = tokenizer(stop_string, return_tensors='pt').input_ids.squeeze().to(USE_DEVICE)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        return torch.all((self.stop_token_seq == input_ids[0][-len(self.stop_token_seq):])).item()


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def inference(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(USE_DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            inputs,
            max_new_tokens=500,
            stopping_criteria=StoppingCriteriaList([StopOnTokens(), StopOnString("Query:", tokenizer), StopOnString("Query: ", tokenizer)]),
            do_sample=True,
            temperature=0.5, # default: 1.0
            top_k=50, # default: 50
            top_p=0.5, # default: 1.0
        )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
    completion = generated_text[len(prompt):]
    return completion


def post_process_completion(completion):
    no_stop_word = completion.split("Query:")[0].rstrip()
    final_sentence = no_stop_word.split('. ')[-1].rstrip(".")
    should_be_code_snippet = re.split('The answer is|The answer is:', final_sentence)[-1].strip()
    no_back_quotes = should_be_code_snippet.replace("`", "").rstrip('\n').rstrip().lstrip('\n').lstrip()
    return no_back_quotes

'''
based on retrieve_for_query in synthesize/synthesize_all.py
'''
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
        current_retrieval = ""
        for f, d in zip(f_list, d_list):
            current_retrieval += f + "    "
            current_retrieval += d.strip().rstrip(":") + "    "
        retrievals.append(current_retrieval)
    # return retrievals
    return {q_id: retrieve for q_id, retrieve in zip(query_ids, retrievals)}


def code_generation_inference_task(model, tokenizer, icl_exemplars=2, with_query_enhancement=True, with_retrieval=True, with_rationale=True, save_as="mosaic-chat"):
    '''
    icl_exemplars: number of example (query, answer)'s to include before the test query
    with_query_enhancement: include the enhancement to the query in the test query and all icl exemplars?
    with_retrieval: include retrieved code documentation in the test query and all icl exemplars?
    with_rationale: include rationale before answer in all icl exemplars?
    '''
    
    query_id_to_oracle = {}
    with open(TEST_ORACLE, "r") as fin:
        full_oracle = json.load(fin)
        for exemplar in full_oracle:
            query_id_to_oracle[exemplar["question_id"]] = (exemplar["nl"], exemplar["cmd"])
    query_id_to_retrieval = retrieve_for_query(query_id_to_oracle.keys(), truncate=True)

    train_rationales = {}
    with open(HUMAN_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            train_rationales[exemplar["question_id"]] = (exemplar["query"], exemplar["retrieval"], exemplar["rationale"])
    with open(GPT3_RATIONALES, "r") as fin:
        for line in fin:
            exemplar = json.loads(line)
            train_rationales[exemplar["question_id"]] = (exemplar["query"], exemplar["retrieval"], exemplar["rationale"])

    assistant = "You are an expert language model in code generation. "
    assistant += f"Come up with {'a rationale' if with_rationale else 'an answer'} for a code generation problem under the following specification. "
    assistant += "Given a query for a coding task"
    if with_retrieval:
        assistant += " and a list of code documentation"
    assistant += ", please "
    if with_rationale:
        assistant += "reason through the provided documentation to arrive at the answer code and "
    assistant += "print the answer at the end of the output. "
    if icl_exemplars > 0:
        assistant += f"{'An example' if icl_exemplars==1 else 'A few examples'} of (query, {'relevant documentation, ' if with_retrieval else ''}{'rationale' if with_rationale else 'answer'}) have been provided. "
        assistant += "Please emulate the format of the provided examples and return only the final rationale for the final query.\n\n"
    else:
        assistant += "Your response should state \"The answer is \" followed by the correct code snippet.\n\n"

    to_json = str(ROOT_DIR / f'llama/completions/{save_as}/{icl_exemplars}_{with_retrieval}_{with_rationale}.json')
    res = []
    for query_id in query_id_to_oracle:

        prompt = assistant

        for _ in range(icl_exemplars):
            icl_id, (icl_query, icl_retrieval, icl_rationale) = random.choice(tuple(train_rationales.items()))
            prompt += f"Query: {icl_query if with_query_enhancement else icl_query.split('. ')[0]}\n"
            if with_retrieval:
                prompt += f"Relevant code documentation: {icl_retrieval}\n"
            if with_rationale:
                prompt += f"Rationale: {icl_rationale}\n\n"
            else:
                prompt += f"Answer: {icl_rationale.split('. ')[-1]}\n\n"
            
            print(prompt)
            return
        return
        test_query, oracle_answer = query_id_to_oracle[query_id]
        test_retrieval = query_id_to_retrieval[query_id]
        # NO TEST QUERY ENHANCEMENT YET!!
        prompt += f"Query: {test_query}.\n"
        if with_retrieval:
            prompt += f"Relevant code documentation: {test_retrieval}\n"
        prompt += f"{'Rationale: ' if with_rationale else 'Answer: '}"
        completion = inference(model, tokenizer, prompt)
        test_answer = post_process_completion(completion)
        print(f" || For query {query_id},\n || LLM answered {test_answer}\n || and oracle answer was {oracle_answer}.\n\n")
        print(f" || For query {query_id},\n || LLM answered {test_answer}\n || and oracle answer was {oracle_answer}.\n\n", file=sys.stderr)

        res.append((query_id, test_answer, oracle_answer))

    with open(to_json, 'w') as fout:
        fout.write(
            json.dumps({
                f"({icl_exemplars}, {with_retrieval}, {with_rationale})": res
            })
        )


def ablation_array():
    model, tokenizer = load_model(MOSAIC_CHAT_AT)

    opt = parse_opt()
    job_array = opt.exparray

    if job_array == 1:
        code_generation_inference_task(model, tokenizer, icl_exemplars=0, with_retrieval=False, with_rationale=False)
    if job_array == 2:
        code_generation_inference_task(model, tokenizer, icl_exemplars=0, with_rationale=False)
    if job_array == 3:
        code_generation_inference_task(model, tokenizer, icl_exemplars=2, with_retrieval=False, with_rationale=False)
    if job_array == 4:
        code_generation_inference_task(model, tokenizer, icl_exemplars=2, with_rationale=False)
    if job_array == 5:
        code_generation_inference_task(model, tokenizer, icl_exemplars=2, with_retrieval=False)
    if job_array == 6:
        code_generation_inference_task(model, tokenizer, icl_exemplars=2)


def quick_merge(saved_as="mosaic-chat"):
    json_location = str(ROOT_DIR / f'llama/completions/{saved_as}')
    all_ablations = json_location + "/res.json"
    if Path(all_ablations).is_file():
        return
    ablation_pairs = glob.glob(json_location + '/*.json')
    print(ablation_pairs)
    res = {}
    for pair in ablation_pairs:
        dict_ablation_and_output = json.load(open(pair, 'r'))
        for ablation, output in dict_ablation_and_output.items():
            break
        print(ablation)
        res[ablation] = output
    with open(all_ablations, 'w') as fout:
        fout.write(json.dumps(res))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--exparray', type=int, default=0, help='job array id from slurm parallelization')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    # ablation_array()
    assistant = "You are an expert language model in code generation. "
    assistant += f"Come up with a rationale for a code generation problem under the following specification. "
    assistant += "Given a query for a coding task and a list of code documentation, "
    assistant += "please reason through the provided documentation to arrive at the answer code and "
    assistant += "print the answer at the end of the output. "
    assistant += "The final sentence in your response should state \"The answer is \" followed by the correct code snippet.\n\n"

    print(assistant)

    # with open(str(ROOT_DIR / "llama/rationale_icl_example.txt"), 'r') as fin:
    #     prompt = fin.read()
    # print("Prompt:")
    # pprint(prompt)
    # print('\n\n')
    # print("Completion:")
    # pprint(inference(prompt, model_name=MOSAIC_CHAT_AT))