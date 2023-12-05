# Under development

from eval import DSTEvaluator
import argparse
import os
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import pandas as pd
from datasets import (
    load_dataset,
    Dataset,
)
from functools import partial 

INSTRUCTION_TEMPLATE = (
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    )

def preprocess_prompt(examples, tokenizer):
    instructions = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]

        instruction = INSTRUCTION_TEMPLATE.format_map({"instruction": instruction, "input": input_text})
        instructions.append(instruction)

    return {
        "input": instructions,
        "output": examples["output"],
    }

parser = argparse.ArgumentParser()

parser.add_argument("--adapter_path", type=str, default=None)
parser.add_argument("--base_model_name", type=str, default="gpt2")
parser.add_argument("--eval_data_path", type=str, required=True)
parser.add_argument("--dataset_format", type=str, default="single_qa")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="./temp/")
args = parser.parse_args()

args.output_dir = os.path.join("./data/beam")

# initiate evaluator
TAKE_FROM_PCT = 0.3
TAKE_TO_PCT = 0.5

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    )

tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side="left", use_fast=True) #, use_fast=False

tokenizer.pad_token = tokenizer.unk_token 

gen_config = GenerationConfig(
        max_new_tokens=32, # a large enough value
        do_sample=False,
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

# load dataset
eval_data = load_dataset("json", data_files={"train": args.eval_data_path})["train"]
eval_data = eval_data.select(range(int(len(eval_data)*0.3), int(len(eval_data)*0.6)))
preprocess_prompt_partial = partial(preprocess_prompt, tokenizer=tokenizer)
eval_data = eval_data.map(preprocess_prompt_partial, batched=True, batch_size=100, remove_columns=eval_data.column_names, num_proc=8)

evaluator = DSTEvaluator(
        eval_data_file_or_path=eval_data,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
    )

evaluator.prepare(model, tokenizer, gen_config=gen_config)

results = evaluator.inference_single_qa_beam_upper_bound(
        data=evaluator.eval_data,
        ).to_pandas()

results.to_csv(os.path.join(args.output_dir, "beam_results.csv"), index=False)