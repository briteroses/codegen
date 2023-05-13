import os
import torch
import numpy as np
import json
import random
import transformers
from datasets import Dataset

import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from pathlib import Path
from pprint import pprint

from utils.load import load_model

DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]

MOSAIC_CHAT_AT = 'mosaicml/mpt-7b-chat'

FOR_FINETUNING_AT = str(ROOT_DIR / "synthesize/data/for_finetuning.json")
SAVE_FINETUNED = str(ROOT_DIR / "llama/models/mosaic-chat-lora/")


def lora_ify(model_name):
    model, tokenizer = load_model(model_name)

    # for param in model.parameters():
    #     param.requires_grad = False
    #     if param.ndim == 1:
    #         param.data = param.data.to(torch.float32)

    model.enable_input_require_grads()

    # config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     # target_modules=["q_proj", "v_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, config)

    return model, tokenizer


def finetune(model_name):
    model, tokenizer = lora_ify(model_name)
    with open(FOR_FINETUNING_AT, "r") as fin:
        for_finetuning = Dataset.from_dict(json.load(fin))

    train_split = for_finetuning["train"].train_test_split(
        test_size=0, shuffle=True
    )
    train_dataset = train_split["train"].map(lambda x: tokenizer(x, return_tensors="pt"))

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            output_dir=SAVE_FINETUNED,
            report_to="none"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()
    model.save_pretrained(SAVE_FINETUNED)


if __name__ == '__main__':
    finetune(MOSAIC_CHAT_AT)