# This util code is heavily adapted from fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
import os
from typing import Dict, Optional, Sequence
from functools import partial
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from utils import first_difference_index
import pdb
from tqdm import tqdm

WORLD_SIZE = os.cpu_count()

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

INSTRUCTION_TEMPLATE = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            )
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    max_len: int = field(
        default=1024, metadata={"help": "Maximum length for each text to pad / truncate"}
    )
    indicator_ids: str = field(
        default="\n\n### Response:\n", metadata={"help": "Indicator for the output"}
    )
    debug_mode: bool = False
    train_pct: float = field(
        default=1.0, metadata={"help": "Percentage of data to keep in train set."}
    )
    eval_pct: float = field(
        default=1.0, metadata={"help": "Percentage of data to keep in train set."}
    )

def preprocess(
    samples,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    indicator_ids: list,
) -> Dict:
    """Preprocess the data."""

    ins_res_list = []
    
    for i in range(len(samples["instruction"])):
        ins = INSTRUCTION_TEMPLATE.format_map(
            {
                "instruction": samples["instruction"][i],
                "input": samples["input"][i],
                "output": samples["output"][i],
            }
        )
        if len(tokenizer(ins)["input_ids"]) <= max_len:
            ins_res_list.append(ins)

    input_ids = tokenizer(
        ins_res_list,
        return_tensors="pt",
        padding="max_length",
        max_length=max_len,
    ).input_ids
    targets = input_ids.clone()
    # Find the indicator ids in target and make the loss function ignore 
    # the previous context.
    indicator_ids_start_idx = None
    
    for i in range(targets.shape[0]):
        for idx in np.where(targets[i] == indicator_ids[0])[0]:
            # `indicator_ids` is `'### Response:\n'`. Find the last occurence.
            if (
                indicator_ids
                == targets[i, idx : idx + len(indicator_ids)].tolist()
            ):
                indicator_ids_start_idx = idx

        if indicator_ids_start_idx is None:
            warnings.warn(
                f"Could not find response key `{tokenizer.decode(indicator_ids)}` in the "
                f'following instance: {tokenizer.decode(targets[i])} '
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            targets[i, :] = IGNORE_TOKEN_ID
        else:
            indicator_ids_end_idx = indicator_ids_start_idx + len(indicator_ids)

            # Make pytorch loss function ignore all tokens up through the end of the response key
            targets[i, :indicator_ids_end_idx] = IGNORE_TOKEN_ID
        
    # Lastly, change the pad token ids to -100 to avoid calculating loss on it.
    targets[targets == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=(input_ids != tokenizer.pad_token_id),
    )

def preprocess_unlikelihood(samples,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int):
    """Preprocess raw dataset for unlikelihood training."""

    TEMP_UNLIKELY_CANDIDATES = 5

    input_ids = []
    targets = []
    attention_mask = []

    unlikely_input_ids = []
    unlikely_targets = []
    unlikely_attention_mask = []

    for i in tqdm(range(len(samples["input"]))):
        instruction_tokenized_ids = tokenizer(samples["input"][i]).input_ids[:-1]
        try:
            target_tokenized_ids = tokenizer("\n" + samples["output"][i]).input_ids[3:]
        except:
            pdb.set_trace()
        
        candidate_input_ids = []
        candidate_targets = []
        candidate_attention_masks = []

        for candidate in samples["unlikely_candidates"][i]:
            candidate_tokenized_ids = tokenizer("\n" + candidate).input_ids[3:]
            diff_idx = first_difference_index(target_tokenized_ids, candidate_tokenized_ids)
            
            candidate_input_ids_temp = instruction_tokenized_ids + candidate_tokenized_ids + [tokenizer.pad_token_id] * (max_len - len(instruction_tokenized_ids) - len(candidate_tokenized_ids))
            if len(candidate_input_ids_temp) > max_len:
                break
            candidate_input_ids.append(candidate_input_ids_temp)
            if diff_idx == -1:
                candidate_targets.append([-100] * len(candidate_input_ids_temp))
            else:
                candidate_targets.append([-100] * (len(instruction_tokenized_ids)+diff_idx) + candidate_tokenized_ids[diff_idx:] + [-100] * (max_len - len(instruction_tokenized_ids) - len(candidate_tokenized_ids)))
            candidate_attention_masks.append([1] * (len(instruction_tokenized_ids) + len(candidate_tokenized_ids)) + [0] * (max_len - len(instruction_tokenized_ids) - len(candidate_tokenized_ids)))

        if len(candidate_input_ids) != TEMP_UNLIKELY_CANDIDATES:
            print("Not enough candidates for this sample, because of exceeding max_len or the input unlikely candidates are broken. Skipping...")
            print(samples["unlikely_candidates"][i])
            continue
        
        sample_input_ids = instruction_tokenized_ids + target_tokenized_ids + [tokenizer.pad_token_id] * (max_len - len(instruction_tokenized_ids) - len(target_tokenized_ids))
        sample_targets = [-100] * len(instruction_tokenized_ids) + target_tokenized_ids + [tokenizer.pad_token_id] * (max_len - len(instruction_tokenized_ids) - len(target_tokenized_ids))
        sample_attention_mask = [1] * (len(instruction_tokenized_ids) + len(target_tokenized_ids)) + [0] * (max_len - len(instruction_tokenized_ids) - len(target_tokenized_ids))

        input_ids.append(sample_input_ids)
        targets.append(sample_targets)
        attention_mask.append(sample_attention_mask)

        unlikely_input_ids.append(candidate_input_ids)
        unlikely_targets.append(candidate_targets)
        unlikely_attention_mask.append(candidate_attention_masks)

    input_ids = torch.tensor(input_ids) #(dataset_size, max_len)
    targets = torch.tensor(targets)
    attention_mask = torch.tensor(attention_mask)

    unlikely_input_ids = torch.tensor(unlikely_input_ids) #(dataset_size, 5, max_len)
    unlikely_targets = torch.tensor(unlikely_targets)
    unlikely_attention_mask = torch.tensor(unlikely_attention_mask)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
        unlikely_input_ids=unlikely_input_ids,
        unlikely_targets=unlikely_targets,
        unlikely_attention_mask=unlikely_attention_mask,
    )

class UnlikelihoodDataset(Dataset):
    """Dataset for unlikelihood training."""

    def __init__(self, raw_data_path, tokenizer: transformers.PreTrainedTokenizer, max_len, debug):
        super(UnlikelihoodDataset, self).__init__()
        raw_data = self._load(raw_data_path, debug)

        rank0_print("Formatting inputs...")
        data_dict = preprocess_unlikelihood(raw_data, tokenizer, max_len)
        warnings.warn(f"{len(raw_data['input']) - len(data_dict['input_ids'])} data points removed because they exceed max_len={max_len}.")
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.unlikely_input_ids = data_dict["unlikely_input_ids"]
        self.unlikely_targets = data_dict["unlikely_targets"]
        self.unlikely_attention_mask = data_dict["unlikely_attention_mask"]

    def __len__(self):
        return len(self.input_ids)
    
    def _load(self, raw_data_path, debug):
        raw_data = pd.read_csv(raw_data_path, keep_default_na=False,na_values=['NaN'])

        # HACK: hard fix of some text formatting issues in csv
        raw_data["unlikely_candidates"] = raw_data["single_qa_beam_candidates"].apply(lambda x: x[2:-2].replace("\n", "").replace('"', "'").split("' '"))
        if debug:
            raw_data = raw_data.iloc[:100]
        return raw_data.to_dict()
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            unlikely_input_ids=self.unlikely_input_ids[i],
            unlikely_targets=self.unlikely_targets[i],
            unlikely_attention_mask=self.unlikely_attention_mask[i],
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len, indicator_ids):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        data_dict = preprocess(raw_data, tokenizer, max_len, indicator_ids)
        warnings.warn(f"{len(raw_data['input']) - len(data_dict['input_ids'])} data points removed because they exceed max_len={max_len}.")
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    # XXX NOTE: Why do we use the lazy version where we preprocess only when we get it?
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len, indicator_ids):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        keys = raw_data.keys()
        self.raw_data = [dict(zip(keys, [[v] for v in values])) for values in zip(*raw_data.values())]
        self.cached_data_dict = {}
        self.max_len = max_len
        self.indicator_ids = indicator_ids

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(self.raw_data[i], self.tokenizer, self.max_len, self.indicator_ids)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )

    # HACK: The first three tokens are [empty string, \n, \n]. 
    #       There is something wrong with the llama tokenizer 
    #       and this is a workaround.
    output_indicator_ids = tokenizer.encode(data_args.indicator_ids, add_special_tokens=False)[3:] 

    rank0_print("Loading data...")

    jsonl_files = {
            "train": os.path.join(data_args.data_path, "train.jsonl"),
            "validation": os.path.join(data_args.eval_data_path, "val.jsonl"),
        }
    data = load_dataset("json", data_files=jsonl_files)

    data["train"], data["validation"] = data["train"].shuffle(seed=42), data["validation"].shuffle(seed=42)
    data["train"], data["validation"] = data["train"].select(range(int(len(data["train"]) * data_args.train_pct))), data["validation"].select(range(int(len(data["validation"]) * data_args.eval_pct)))

    print("Number of training examples:", len(data["train"]))
    print("Number of validation examples:", len(data["validation"]))
    
    if data_args.debug_mode:
        data["train"], data["validation"] = data["train"].select(range(100)), data["validation"].select(range(50))
        
    train_data, val_data = data["train"].to_dict(), data["validation"].to_dict()
    del data

    train_dataset = dataset_cls(train_data, tokenizer=tokenizer, max_len=data_args.max_len, indicator_ids=output_indicator_ids)
    rank0_print("Train dataset processed.")
    eval_dataset = dataset_cls(val_data, tokenizer=tokenizer, max_len=data_args.max_len, indicator_ids=output_indicator_ids)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def make_supervised_data_module_unlikelihood(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    
    # HACK: The first three tokens are [empty string, \n, \n]. 
    #       There is something wrong with the llama tokenizer 
    #       and this is a workaround.
    output_indicator_ids = tokenizer.encode(data_args.indicator_ids, add_special_tokens=False)[3:] 

    rank0_print("Loading data...")
    # train
    train_dataset = UnlikelihoodDataset(data_args.data_path, tokenizer=tokenizer, max_len=data_args.max_len, debug=data_args.debug_mode)

    # validation
    jsonl_files = {
            "validation": os.path.join(data_args.eval_data_path, "val.jsonl"),
        }
    data = load_dataset("json", data_files=jsonl_files)
    eval_dataset = data["validation"].shuffle(seed=42)
    eval_dataset = eval_dataset.select(range(int(len(eval_dataset) * data_args.eval_pct)))
    print("Number of validation examples:", len(eval_dataset))
    if data_args.debug_mode:
        eval_dataset = eval_dataset.select(range(50))
        
    eval_dataset = eval_dataset.to_dict()
    eval_dataset = dataset_cls(eval_dataset, tokenizer=tokenizer, max_len=data_args.max_len, indicator_ids=output_indicator_ids)
    del data

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

if __name__ == "__main__":
    print("Debugging dataset_utils.py")
    print("Number of CPU cores:", WORLD_SIZE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "/scratch/bc3088/LF-research/llama/hf-models/llama-2-7b-chat",
        padding_side="right",
        use_fast=False,
        add_eos_token=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_args = DataArguments(
        data_path="./data/MultiWOZ_2.2_instruction/",
        eval_data_path="./data/MultiWOZ_2.2_instruction/",
        lazy_preprocess=False,
        max_len=640,
        indicator_ids="\n\n### Response:\n",
    )

    data_module = make_supervised_data_module(tokenizer, data_args)
    print(data_module["train_dataset"][0])
    print("Done")