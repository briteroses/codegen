import torch
import numpy as np
import transformers

from pathlib import Path
from pprint import pprint

from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from transformers import LlamaForCausalLM, LlamaTokenizer


DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
USE_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

ROOT_DIR = Path(__file__).resolve().parents[1]
LLAMA_PRETRAINED_AT = str(ROOT_DIR / "llama/llama_hf_7b")


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def inference(prompt):
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_PRETRAINED_AT, low_cpu_mem_usage=True)
    model = LlamaForCausalLM.from_pretrained(LLAMA_PRETRAINED_AT, low_cpu_mem_usage=True)
    model.half().to(USE_DEVICE)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generate_ids = model.generate(inputs.input_ids, stopping_criteria=StoppingCriteriaList([StopOnTokens()]), max_length=1024)
    decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return decoded


if __name__ == '__main__':
    # prompt = "Use the provided information to answer a single no explanation code command. For example:\nInstruction: perform a bruteforce attack (mode 3) with the default hashcat mask\nAnswer: hashcat --hash-type {{hash_type_id}} --attack-mode {{3}} {{hash_value}}\n\n\"\"\"\"\nipcrm - remove certain IPC resources\nipcrm removes System V inter-process communication (IPC) objects and associated data structures from the system. In order to delete such objects, you must be superuser, or the creator or owner of the object.\nSystem V IPC objects are of three types: shared memory, message queues, and semaphores. Deletion of a message queue or semaphore object is immediate (regardless of whether any process still holds an IPC identifier for the object). A shared memory object is only removed after all currently attached processes have detached (shmdt(2)) the object from their virtual address space.\nTwo syntax styles are supported. The old Linux historical syntax specifies a three-letter keyword indicating which class of object is to be deleted, followed by one or more IPC identifiers for objects of this type.\nThe SUS-compliant syntax allows the specification of zero or more objects of all three types in a single command line, with objects specified either by key or by identifier (see below). Both keys and identifiers may be specified in decimal, hexadecimal (specified with an initial '0x' or '0X'), or octal (specified with an initial '0').\n-M, --shmem-key shmkey Remove the shared memory segment created with shmkey after the last detach is performed.\n-Q, --queue-key msgkey Remove the message queue created with msgkey.\n-q, --queue-id msgid Remove the message queue identified by msgid.\n-S, --semaphore-key semkey Remove the semaphore created with semkey.\nIn its first Linux implementation, ipcrm used the deprecated syntax shown in the second line of the SYNOPSIS. Functionality present in other *nix implementations of ipcrm has since been added, namely the ability to delete resources by key (not just identifier), and to respect the same command-line syntax. For backward compatibility the previous syntax is still supported.\n\"\"\"\n\nInstruction: delete an ipc queue by key\nAnswer: "
    # prompt = "John and Mary went to the beach. Who did John take pictures for?"
    with open(str(ROOT_DIR / "llama/rationale_icl_example.txt"), 'r') as fin:
        prompt = fin.read()
    print("Prompt:")
    pprint(prompt)
    print('\n\n')
    print("Completion:")
    pprint(inference(prompt))