PRE_SEQ_LEN=512
CHECKPOINT=P_tuning-chatglm3-6b-pt-1024-2e-2
STEP=400
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_predict \
    --validation_file data/dev.json \
    --test_file data/dev.json \
    --overwrite_cache \
    --prompt_column prompt \
    --response_column output \
    --model_name_or_path ../models/chatglm3-6b/ \
    --ptuning_checkpoint ./saves/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./saves/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
