CUDA_VISIBLE_DEVICES=0 python export_model.py \
    --model_name_or_path ../models/Qwen2-7B-Instruct/ \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --export_dir ../models/merge_Qwen2-7B-lora/ \
    --adapter_name_or_path ./saves/Qwen2-7B-lora/chinese_medical/checkpoint-800/
