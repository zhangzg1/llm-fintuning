PRE_SEQ_LEN=1024
LR=2e-2
RUNNING_STEP=400
NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --do_train \
    --train_file data/train.json \
    --validation_file data/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column prompt \
    --response_column output \
    --overwrite_cache \
    --model_name_or_path ../models/chatglm3-6b/ \
    --output_dir ./saves/P_tuning-chatglm3-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps $RUNNING_STEP \
    --logging_steps 2 \
    --save_steps $RUNNING_STEP \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \

