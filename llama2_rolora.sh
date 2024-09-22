CUDA_VISIBLE_DEVICES=0 python src/train.py \
--stage sft \
--do_train True \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--rotate_down_proj \
--rotate_mode 'hadamard' \
--finetuning_type lora \
--template default \
--dataset alpaca_gpt4_en \
--cutoff_len 1024 \
--learning_rate 0.0001 \
--num_train_epochs 3.0 \
--max_samples 100000 \
--per_device_train_batch_size 8 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 10 \
--save_steps 5000 \
--warmup_ratio 0.01 \
--val_size 0.1  \
--per_device_eval_batch_size 16 \
--evaluation_strategy steps \
--eval_steps 5000 \
--optim adamw_torch \
--report_to wandb \
--output_dir saves/llama2-7b-rolora-qv-r8 \
--fp16 True \
--lora_rank 8 \
--lora_alpha 16 \
--lora_dropout 0 \
--lora_target q_proj.module,v_proj.module \
--plot_loss True \
--load_best_model_at_end
