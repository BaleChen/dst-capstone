bash wandb_config.sh # ignore this or use your own wandb config

deepspeed src/fc_train_lora.py \
    --model_name_or_path /scratch/bc3088/LF-research/llama/hf-models/llama-2-7b-chat \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --data_path ./data/MultiWOZ_2.2_instruction/ \
    --eval_data_path ./data/MultiWOZ_2.2_instruction/ \
    --bf16 False \
    --fp16 True \
    --tf32 False \
    --output_dir ./checkpoints \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 2048 \
    --q_lora True \
    --deepspeed ./../FastChat/playground/deepspeed_config_s2.json \
    --report_to wandb \
    --run_name debug \
    # --max_steps 1000 \
    # --debug_mode \

