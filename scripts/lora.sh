bash scripts/wandb_config.sh;

torchrun --nproc_per_node $1 --nnode 1 src/fc_train_lora.py \
    --model_name_or_path /scratch/bc3088/LF-research/llama/hf-models/llama-2-7b-chat \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --data_path ./data/MultiWOZ_2.2_instruction/ \
    --eval_data_path ./data/MultiWOZ_2.2_instruction/ \
    --bf16 False \
    --fp16 False \
    --tf32 False \
    --output_dir ./checkpoints \
    --num_train_epochs 4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --load_best_model_at_end True \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --q_lora True \
    --report_to wandb \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --train_pct 0.3 \
    --eval_pct 0.3 \
    # --checkpoint_dir ./checkpoints/2023-10-07-17:03_llama-2-7b-chat_pdbs16_lr1e-05 \
    # --debug_mode \
    # --max_steps 1000 \
    # --deepspeed ./../FastChat/playground/deepspeed_config_s2.json \

    # Need to specify ddp_find_unused_parameters to avoid a bug in torchrun and gradient checkpointing