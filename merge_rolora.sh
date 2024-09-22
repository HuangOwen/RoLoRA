CUDA_VISIBLE_DEVICES=0 python src/export.py \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--rotate_down_proj \
--rotate_mode 'hadamard' \
--adapter_name_or_path saves/llama2-7b-rolora-qv-r8/ \
--template default \
--finetuning_type lora \
--export_dir models/llama2-7b-rolora-qv-r8 \
--export_size 2 \
--export_device cpu \
--export_legacy_format false 


