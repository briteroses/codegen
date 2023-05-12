import os
import sys
import json
import torch
import numpy as np
import transformers

from pathlib import Path
from pprint import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[2]

LLAMA_AT = str(ROOT_DIR / "llama/llama_hf_7b")
ALPACA_7B_AT = str(ROOT_DIR / "llama/alpaca")
MOSAIC_INSTRUCT_AT = 'mosaicml/mpt-7b-instruct'
MOSAIC_CHAT_AT = 'mosaicml/mpt-7b-chat'
ALPACA_13B_AT = None

ALL_MODEL_NAMES = [LLAMA_AT, ALPACA_7B_AT, MOSAIC_INSTRUCT_AT, MOSAIC_CHAT_AT, ALPACA_13B_AT]


def load_model(model_name=LLAMA_AT):
    '''
    model_name: string for loading weights for huggingfce transformers. can be remote or local.
                only the model names in global ALL_MODEL_NAMES are supported.
    '''

    assert model_name in ALL_MODEL_NAMES, "not supported; please pick a supported model from ALL_MODEL_NAMES instead"

    if model_name == LLAMA_AT:
        tokenizer = LlamaTokenizer.from_pretrained(LLAMA_AT, low_cpu_mem_usage=True)
        model = LlamaForCausalLM.from_pretrained(LLAMA_AT, low_cpu_mem_usage=True).half().to(USE_DEVICE)
    elif model_name == ALPACA_7B_AT:
        tokenizer = LlamaTokenizer.from_pretrained(ALPACA_7B_AT)
        model = LlamaForCausalLM.from_pretrained(
            ALPACA_7B_AT,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            cache_dir="cache"
        ).half().to(USE_DEVICE)
    elif model_name == MOSAIC_INSTRUCT_AT:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'mosaicml/mpt-7b-instruct',
            trust_remote_code=True
        ).half().to(USE_DEVICE)
    elif model_name == MOSAIC_CHAT_AT:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            'mosaicml/mpt-7b-chat',
            trust_remote_code=True
        ).half().to(USE_DEVICE)
    elif model_name == ALPACA_13B_AT:
        raise NotImplementedError

    return model, tokenizer