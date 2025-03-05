import os.path
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model


# 在原模型中插入Lora层，得到新的Lora模型
def build_peft_model(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,       # 训练模式
        r=16,                       # Lora低秩矩阵中rank的大小
        lora_alpha=32,              # 比例因子，用于在前向传播中将Lora参数以一定的缩放比例应用于模型之中
        lora_dropout=0.2            # 在训练过程中丢弃Lora层中神经元的比例
    )
    return get_peft_model(model, config)


# 构建Lora微调的超参数，也就是设置Lora微调是的一些超参数
def build_train_arguments(output_path):
    return TrainingArguments(
        output_dir=output_path,                         # 输出模型的保存目录
        per_device_train_batch_size=4,                  # 每个设备（如每个GPU）的训练批次大小
        gradient_accumulation_steps=4,                  # 梯度累计的步骤数，相对于增加批次大小
        log_level="info",                               # 日志级别
        logging_steps=200,                              # 每隔多少个步骤记录一次日志
        logging_first_step=True,                        # 是否在训练的第一步就记录日志
        logging_dir=os.path.join(output_path, "logs"),  # 设置日志的保存目录
        num_train_epochs=5,                             # 训练的总轮数
        eval_strategy="steps",                          # 设置训练期间的评估验证策略，可选值有steps、epoch、no
        eval_on_start=False,                            # 在训练开始时就进行模型评估
        eval_steps=200,                                 # 设置评估的步数，与保存步数一致
        save_steps=200,                                 # 每隔多少步就保存一次模型
        learning_rate=1e-4,                             # 学习率大小
        lr_scheduler_type="cosine",                     # 学习率调度器，这里使用余弦退火调度器cosine
        warmup_ratio=0.05,                              # 学习率预热比例，这里表示前5%的steps用于预热
        save_on_each_node=True,                         # 分布式训练时是否在每个节点上都保存checkpoint，用于特定节点失败时从指定点恢复训练
        load_best_model_at_end=True,                    # 在训练结束时加载最佳模型
        remove_unused_columns=False,                    # 是否移除数据集中模型训练未使用到的列，以减少内存使用
        dataloader_drop_last=True,                      # 抛弃最后一批迭代数据（数据可能不满足一批，会影响训练效果）
        gradient_checkpointing=True                     # 启用梯度检查点以节省内存
    )


# 构建Lora微调训练器，并将微调训练需要的超参数传入进去
def build_trainer(model, tokenizer, args, train_dataset, eval_dataset):
    if args.gradient_checkpointing:
        model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),  # 控制如何将原始数据合并成批次（batch）
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]              # 早停回调，指定多少次评估后没改善就提前停止训练
    )
