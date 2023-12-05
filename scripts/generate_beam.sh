python src/generate_beam_data.py \
--eval_data_path ./data/MultiWOZ_2.2_instruction/train.jsonl \
--base_model_name /scratch/bc3088/capstone/dst-capstone/checkpoints/2023-10-27-18:02_llama-2-7b-chat_pdbs16_lr1e-05/checkpoint-6500_merged \
--batch_size 4 \
--num_beams 5