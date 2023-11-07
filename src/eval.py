# import checkdst
from datasets import (
    load_dataset,
    Dataset,
)
import pandas as pd
from functools import partial 
import warnings
import torch
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from fastchat.model import (
    load_model
)

from tqdm import tqdm
import pdb
import argparse
import time
import os
import json

INSTRUCTION_TEMPLATE = (
                "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )

def preprocess_prompt(examples, tokenizer, dataset_format="single_qa"):
    instructions = []
    for i in range(len(examples["dialogue_turn_id"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]

        instruction = INSTRUCTION_TEMPLATE.format_map({"instruction": instruction, "input": input_text})
        instructions.append(instruction)

    return {
        "dialogue_turn_id": examples["dialogue_turn_id"],
        "input": instructions,
        "output": examples["output"],
        "slot_name": examples["slot_name"],
        "context": examples["input"],
    }

class DSTEvaluator():
    def __init__(self, eval_data_path=None, dataset_format="single_qa", batch_size=8, gen_config=None):
        self.dataset_format = dataset_format
        self.batch_size = batch_size
        self.eval_data_path = eval_data_path
        self.evaluation_ready = False
    
    def prepare(self, model, tokenizer, gen_config=None):
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        if self.tokenizer.padding_side != "left":
            warnings.warn(
                "The tokenizer padding side is not set to left. This may lead to errors when using the model."
            )
        if self.tokenizer.add_eos_token:
            self.tokenizer.add_eos_token = False
            warnings.warn(
                "The tokenizer.add_eos_token is turned off for tokenizing the prompt."
            )
        if gen_config is None:
            warnings.warn("No generation config is provided. Using default greedy search decoding config.")
            self.gen_config = GenerationConfig(
                max_new_tokens=64, # a large enough value
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        else:
            self.gen_config = gen_config

        if self.eval_data_path:
            self.eval_data = self._load_and_process_data(self.eval_data_path)
            self.eval_data_prefix = self.eval_data_path.split("/")[-1].split(".")[0]
        else:
            self.eval_data = None
            self.eval_data_prefix = "temp"

        self.evaluation_ready = True

    def evaluate_wrapped(
        self,
        eval_data=None,
        save_results=False,
        output_dir="./temp/",
    ):
        if not self.evaluation_ready:
            raise Exception("Please load the model and tokenizer first using load_model_and_tokenizer().")
        # deal with what eval data to use
        if eval_data is None and self.eval_data is not None:
            eval_data = self.eval_data
        elif eval_data is None and self.eval_data is not None:
            raise Exception("No eval data is provided. Please either specify eval_data_path when defining DSTEvaluator or provide eval_data in evaluate().")
        elif eval_data and self.eval_data:
            warnings.warn(
                f"eval_data is provided while the DSTEvaluator already have eval_data loaded. Using the eval_data provided in evaluate() function. Please check if the provided format is {self.dataset_format}"
            )
            self.eval_data_prefix = "temp"
        
        if self.dataset_format == "single_qa":
            generations = self.inference_single_qa(data=eval_data, output_dir=output_dir)
            results = self._postprocess_for_eval(generations)
            if save_results:
                # Save a dictionary to json
                with open(os.path.join(output_dir, f"{self.eval_data_prefix}_results.json"), "w") as f:
                    json.dump(results, f)
            stats = self.compute_metrics(results)
            return stats

        elif self.dataset_format == "json_output":
            raise NotImplementedError

        else:
            raise NotImplementedError("Dataset format {} not implemented".format(self.dataset_format))
    
    def _postprocess_for_eval(self, results):
        """Post-process the generation results for evaluation."""

        if type(results) == Dataset:
            eval_df = results.to_pandas()
        elif type(results) != pd.DataFrame:
            raise Exception(f"{type(results)} not supported by compute_metrics().")
        
        # remaining columns:["dialogue_turn_id", "output" (true labels: list), "{dataset_format}_generated_response"]
        pred_col_name = f"{self.dataset_format}_generated_response"
        # clean the rows where both prediction and true label are None
        eval_df = eval_df[eval_df["output"].apply(lambda x : x.tolist() != ["None"]) | (eval_df[pred_col_name] != 'None')]
        
        output_dict = {}
        generated_dict = {}

        # Iterate through each row in the DataFrame
        for _, row in eval_df.iterrows():
            dialogue_turn_id = row['dialogue_turn_id']
            # Convert the "output" and "generated_response" columns to dictionaries
            output = {row["slot_name"]: row['output'].tolist()} 
            generated = {row["slot_name"]: row[pred_col_name]}
            
            # Merge the slot-value pairs into the respective dictionaries for each dialogue turn
            if dialogue_turn_id not in output_dict:
                output_dict[dialogue_turn_id] = {}
            if dialogue_turn_id not in generated_dict:
                generated_dict[dialogue_turn_id] = {}
            
            output_dict[dialogue_turn_id].update(output)
            generated_dict[dialogue_turn_id].update(generated)
        
        # Iterate through each dialogue turn and compile the results into a json
        results = dict()
        for dialogue_turn_id in output_dict.keys():
            results[dialogue_turn_id] = {
                "true_state": output_dict[dialogue_turn_id],
                "pred_state": generated_dict[dialogue_turn_id],
                "context": eval_df[eval_df["dialogue_turn_id"] == dialogue_turn_id]["context"].iloc[0],
                # index 1 is the first row of many duplicate rows.
            }
        return results
    
    def compute_metrics(self, results):
        # loop through all the dictionary items
        total, turn_acc, joint_acc, F1_pred = 0, 0, 0, 0
        precision, recall = 0, 0
        for dialogue_turn_id, result in results.items():
            pred_state, true_state = result["pred_state"], result["true_state"]
            total += 1

            turn_correct, turn_f1, (turn_precision, turn_recall), jga_flag = self.compute_turn_acc_and_f1(pred_state, true_state)

            if jga_flag:
                joint_acc += 1
            F1_pred += turn_f1
            precision += turn_precision
            recall += turn_recall
        
        precision = precision / total
        recall = recall / total
        joint_acc = joint_acc / total
        F1_score = F1_pred / total
        return {"joint_acc": joint_acc, "slot_f1": F1_score, "precision": precision, "recall": recall} # TODO also compute accuracy
    
    def compute_turn_acc_and_f1(self, pred_state, true_state):
        """Compute the turn-level accuracy, precision, recall, and F1 score."""
        # Compute the turn-level accuracy TODO: TRADE compute acc in a diff way
        turn_correct, tp, fp, fn = 0, 0, 0, 0
        for slot, pred_value in pred_state.items():
            true_values = true_state[slot]
            if pred_value in true_values:
                turn_correct += 1
                tp += 1
            elif pred_value == "None":
                fn += 1
            elif true_values == ["None"]:
                fp += 1
        turn_precision = tp / (tp + fp) if tp + fp > 0 else 0
        turn_recall = tp / (tp + fn) if tp + fn > 0 else 0
        turn_f1 = 2 * turn_precision * turn_recall / (turn_precision + turn_recall) if turn_precision + turn_recall > 0 else 0
        
        return turn_correct, turn_f1, (turn_precision, turn_recall), fp+fn == 0

    def inference_single_qa(self, data, output_dir="./temp/"):
        # TODO support multi-gpu inference
        if not self.evaluation_ready:
            raise Exception("Please load the model and tokenizer first using load_model_and_tokenizer().")

        gen_results = []
        for i in tqdm(range(0, len(data), self.batch_size)):
            if i + self.batch_size > len(data):
                batch = data.select(range(i, len(data)))
            else:
                batch = data.select(range(i, i + self.batch_size))
            
            # Using dynamic padding
            input_ids, attention_mask = self.tokenizer(batch["input"], padding=True, truncation=True, return_tensors="pt").values()
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.gen_config,
            )

            gen_results.extend(
                [
                    out.split("### Response:\n")[1].strip() if len(out.split("### Response:\n")) == 2 else "".join(out.split("### Response:\n")[1:]).strip()
                    for out in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ]
            )

        data = data.add_column("single_qa_generated_response", gen_results)
        
        return data
    
    def inference_single_qa_beam_upper_bound(self, data, output_dir="./temp/"):
        gen_results = []
        for i in tqdm(range(0, len(data), self.batch_size)):
            if i + self.batch_size > len(data):
                batch = data.select(range(i, len(data)))
            else:
                batch = data.select(range(i, i + self.batch_size))
            
            # Using dynamic padding
            input_ids, attention_mask = self.tokenizer(batch["input"], padding=True, truncation=True, return_tensors="pt").values()
            input_ids = input_ids.to(self.model.device)
            attention_mask = attention_mask.to(self.model.device)

            assert self.gen_config.num_beams == self.gen_config.num_return_sequences, "num_beams must be equal to num_return_sequences for beam search upper bound inference."

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.gen_config,
            )
                
            flattened_results = [
                    out.split("### Response:\n")[1].strip() if len(out.split("### Response:\n")) == 2 else "".join(out.split("### Response:\n")[1:]).strip()
                    for out in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                ]

            for i in range(0, len(flattened_results), self.gen_config.num_beams):
                gen_results.append(flattened_results[i:i+self.gen_config.num_beams])

        data = data.add_column("single_qa_beam_candidates", gen_results)
        return data

    def inference_multi_qa(self):
        raise NotImplementedError
    
    def _load_and_process_data(self, eval_data_path):
        eval_data = load_dataset("json", data_files={"eval": eval_data_path})["eval"]
        preprocess_prompt_partial = partial(preprocess_prompt, tokenizer=self.tokenizer, dataset_format=self.dataset_format)
        eval_data = eval_data.map(preprocess_prompt_partial, batched=True, batch_size=100, remove_columns=eval_data.column_names, num_proc=8)
        return eval_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model_name", type=str, default="gpt2")
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--dataset_format", type=str, default="single_qa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./temp/")
    args = parser.parse_args()

    # TODO output_dir name auto generator
    args.output_dir = os.path.join("./results/", "_".join(args.base_model_name.split("/")[-2:]))

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        device_map="auto",
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, padding_side="left", use_fast=True) #, use_fast=False

    tokenizer.pad_token = tokenizer.unk_token
    
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_path,
            device_map="auto"
        )
    else:
        model = base_model

    gen_config = GenerationConfig(
        max_new_tokens=64, # a large enough value
        do_sample=False,
        num_beams=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    evaluator = DSTEvaluator(
        eval_data_path=args.eval_data_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
    )
    evaluator.prepare(model, tokenizer, gen_config=gen_config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    stats = evaluator.evaluate_wrapped(save_results=True, output_dir=args.output_dir)

    # write stats to args.output_dir as a .metric file
    with open(os.path.join(args.output_dir, f"{evaluator.eval_data_prefix}.metrics"), "w") as f:
        f.write(str(stats))
