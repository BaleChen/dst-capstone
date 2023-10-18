python src/eval.py \
--eval_data_path ./data/MultiWOZ_2.2_instruction/eval.jsonl \
--base_model_name /scratch/bc3088/capstone/dst-capstone/checkpoints/2023-10-14-14:20_llama-2-7b-chat_pdbs16_lr1e-05/checkpoint-1500_merged \
--output_dir ./temp/2023-10-14-14:20_checkpoint-1500_merged_nb$1/ \
--batch_size 48 \
--num_beams $1 \