from NER_util import process_data, load_model, load_dataset, test_with_batch, test_with_vllm
from NER_lora import build_peft_model, build_train_arguments, build_trainer


def main():
    # 处理原始的训练数据，生成微调的训练数据集
    input_train_path = "/root/dataset/ChineseMedical/medical_train.txt"
    output_train_path = "/root/dataset/ChineseMedical/train_dataset.json"
    process_data(input_train_path, output_train_path)
    # 处理原始的评估数据，生成微调的评估数据集
    input_eval_path = "/root/dataset/ChineseMedical/medical_eval.txt"
    output_eval_path = "/root/dataset/ChineseMedical/eval_dataset.json"
    process_data(input_eval_path, output_eval_path)
    # 处理原始的测试数据，生成模型评测的测试数据集
    input_tset_path = "/root/dataset/ChineseMedical/medical_test.txt"
    output_test_path = "/root/dataset/ChineseMedical/test_dataset.json"
    process_data(input_tset_path, output_test_path)

    model_path = "/root/llm_lora/models/Qwen2_7B_Instruct"
    train_data_path = "dataset/ChineseMedical/train_dataset.json"
    eval_data_path = "dataset/ChineseMedical/eval_dataset.json"
    test_data_path = "dataset/ChineseMedical/test_dataset.json"
    lora_output_path = "saves/medical_lora_model"

    # 加载模型和分词器
    model, tokenizer = load_model(model_path)

    # 在原模型中添加Lora层
    peft_model = build_peft_model(model)
    peft_model.print_trainable_parameters()

    # 设置Lora微调时的超参数
    lora_args = build_train_arguments(lora_output_path)

    # 加载训练、验证数据集
    train_dataset = load_dataset(train_data_path, tokenizer)
    eval_dataset = load_dataset(eval_data_path, tokenizer)

    # 开始Lora微调模型
    trainer = build_trainer(peft_model, tokenizer, lora_args, train_dataset, eval_dataset)
    trainer.train()

    # 在训练结束后会自动加载最佳模型，然后使用测试集对其进行批量测试
    test_with_batch(model, tokenizer, test_data_path, batch_size=16)

    # model_lora = "/root/llm_lora/models/Qwen2_7B_Instruct_lora"
    # model_lora_quantized = "/root/llm_lora/models/Qwen2_7B_Instruct_lora_awq"
    # # 使用vllm大模型加速推理框架对测试数据集进行批次评测
    # test_with_vllm(model_lora, test_data_path, batch_size=16)
    # # 使用量化后的大模型加载vllm框架并进行批次评测
    # test_with_vllm(model_lora_quantized, test_data_path, batch_size=16, quantization="AWQ")


if __name__ == "__main__":
    main()
