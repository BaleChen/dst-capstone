from eval import DSTEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
import torch
import argparse
import os

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
        num_return_sequences=args.num_beams,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    evaluator = DSTEvaluator(
        eval_data_path=args.eval_data_path,
        dataset_format=args.dataset_format,
        batch_size=args.batch_size,
    )
    evaluator.prepare(model, tokenizer, gen_config=gen_config)

    results = evaluator.inference_single_qa_beam_upper_bound(data=evaluator.eval_data).to_pandas()
    
    def check_pred_correct(row):
        candidates = row["single_qa_beam_candidates"].tolist()
        true_labels = row["output"].tolist()
        index = -1
        for l in true_labels:
            try:
                index = candidates.index(l)
            except ValueError:
                continue
        return index
    results["correct_index"] = results.apply(check_pred_correct, axis=1)

    eval_data_prefix = evaluator.eval_data_path.split("/")[-1].split(".")[0]
    results.to_csv(os.path.join(args.output_dir, eval_data_prefix+"_beam_results.csv"), index=False)
