import torch
import numpy as np
import transformers

from pathlib import Path
from pprint import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM, LlamaTokenizer
from accelerate import Accelerator
import accelerate
import time


model = None
tokenizer = None
generator = None

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]
LLAMA_PRETRAINED_AT = str(ROOT_DIR / "llama/llama_hf_7b")
ALPACA_FINETUNED_AT = str(ROOT_DIR / "llama/alpaca")
MOSAIC_AT = 'mosaicml/mpt-7b-instruct'
model = transformers.AutoModelForCausalLM.from_pretrained(
  ,
  trust_remote_code=True
)


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

def load_model(model_name=None, use_pretrained=LLAMA_PRETRAINED_AT):
    '''
    model_name: string for loading remote weights from huggingface transformers.
                Only if None, we load local weights from use_pretrained instead.
    use_pretrained: load local weights. only LLAMA_PRETRAINED_AT and ALPACA_FINETUNED_AT are supported right now.
    ** if model_name is not None, i.e. loading remote weights, then use_pretrained doesn't matter, just keep it at a garbage value
    '''

    global model, tokenizer, generator

    assert use_pretrained in [LLAMA_PRETRAINED_AT, ALPACA_FINETUNED_AT], "only our local llama and alpaca are supported"

    if model_name is None: # default to our local pretrained weights
        if use_pretrained == LLAMA_PRETRAINED_AT:
            tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PRETRAINED_AT, low_cpu_mem_usage=True)
            model = LlamaForCausalLM.from_pretrained(LLAMA_PRETRAINED_AT, low_cpu_mem_usage=True).half().to(USE_DEVICE)
        elif use_pretrained == ALPACA_FINETUNED_AT:
            tokenizer = LlamaTokenizer.from_pretrained(ALPACA_FINETUNED_AT)
            model = LlamaForCausalLM.from_pretrained(
                ALPACA_FINETUNED_AT,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                load_in_8bit=False,
                cache_dir="cache"
            ).half().to(USE_DEVICE)
    else:   # load from huggingface based on name
        tokenizer = 

    generator = model.generate


def inference(prompt):
    if model is None:
        load_model()
    invitation = "Assistant: "
    human_invitation = "Human: "

    fulltext = human_invitation + prompt + "\n\n" + invitation

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.to(USE_DEVICE)
    generated_ids = generator(
        gen_in,
        max_new_tokens=333,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.1,
        temperature=0.5, # default: 1.0
        top_k = 50, # default: 50
        top_p = 1.0, # default: 1.0
        early_stopping=True,
    )
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?
    completion = generated_text[len(fulltext):]
    completion = completion.split(human_invitation)[0].strip()
    return completion

if __name__ == "__main__":
    with open(str(ROOT_DIR / "llama/rationale_icl_example.txt"), 'r') as fin:
        prompt = fin.read()
    print("Prompt:")
    pprint(prompt)
    print('\n\n')
    print("Completion:")
    pprint(inference(prompt))