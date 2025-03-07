import json
import pandas
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import f1_score
from vllm_code.vllm_predict import ChatLLM


# 加载模型和分词器
def load_model(model_path, checkpoint_path='', device='cuda'):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(
        device)
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to(device)
    for param in model.base_model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    return model, tokenizer


# 利用模型进行单次预测生成答案
def predict(model, tokenizer, prompt, device='cuda'):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def build_prompt(content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    return messages


# 利用模型进行批量预测生成答案
def predict_batch(model, tokenizer, contents, device='cuda'):
    prompts = [build_prompt(content) for content in contents]
    text = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(text, padding=True, return_tensors="pt").to(device)
    gen_kwargs = {"max_new_tokens": 1024, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            output = outputs[i, model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
        return responses


# 加载json数据集
def load_json_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return []


# 对数据进行分词并转化为 token_id，最终得到输入、mask、输出对应的 token_id
def data_preprocess(item, tokenizer, max_length=1024):
    system_message = "You are a helpful assistant."
    user_message = item["instruction"] + item["input"]
    assistant_message = item["output"]

    instruction = tokenizer(f"<|im_start|>system\n{system_message}<|im_end|>\n"
                            f"<|im_start|>user\n{user_message}<|im_end|>\n"
                            f"<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(assistant_message, add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = len(input_ids) * [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    return {
        "input_ids": input_ids[:max_length],
        "attention_mask": attention_mask[:max_length],
        "labels": labels[:max_length]
    }


# 加载训练、验证、测试数据集
def load_dataset(data_path, tokenizer):
    data_list = load_json_data(data_path)
    data_set = Dataset.from_pandas(pandas.DataFrame(data_list))
    model_data = data_set.map(lambda x: data_preprocess(x, tokenizer), remove_columns=data_set.column_names)

    return model_data


# 使用测试数据集对模型进行不分批次的评测
def test_without_batch(model, tokenizer, test_path):
    test_dataset = load_json_data(test_path)
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')
    for item in test_dataset:
        prompt = item["instruction"] + item["input"]
        pred_label = predict(model, tokenizer, prompt)
        real_label = item["output"]
        pred_label = pred_label.replace("'", '"')
        real_label = real_label.replace("'", '"')
        real_list = json.loads(real_label)
        try:
            pred_list = json.loads(pred_label)
            if len(pred_list) < len(real_list):
                pred_list += ["O"] * (len(real_list) - len(pred_list))
            elif len(pred_list) > len(real_list):
                pred_list = pred_list[:len(real_list)]
            f1 = f1_score(real_list, pred_list, average='micro')
            f1_score_list.append(f1)
        except json.JSONDecodeError:
            pass
        pbar.update(1)
    pbar.close()
    f1_avg_score = sum(f1_score_list) / len(f1_score_list)
    print(f"f1-score: {f1_avg_score:.4f}")


# 使用测试数据集对模型进行批次的评测
def test_with_batch(model, tokenizer, test_path, batch_size: int = 8):
    test_dataset = load_json_data(test_path)
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')
    for i in range(0, len(test_dataset), batch_size):
        batch_data = test_dataset[i:i + batch_size]
        prompt_batch = [item["instruction"] + item["input"] for item in batch_data]
        pred_label = predict_batch(model, tokenizer, prompt_batch)
        real_label = [item["output"] for item in batch_data]
        for pred_label, real_label in zip(pred_label, real_label):
            pred_label = pred_label.replace("'", '"')
            real_label = real_label.replace("'", '"')
            real_list = json.loads(real_label)
            try:
                pred_list = json.loads(pred_label)
                if len(pred_list) < len(real_list):
                    pred_list += ["O"] * (len(real_list) - len(pred_list))
                elif len(pred_list) > len(real_list):
                    pred_list = pred_list[:len(real_list)]
                f1 = f1_score(real_list, pred_list, average='micro')
                f1_score_list.append(f1)
            except json.JSONDecodeError:
                pass
        pbar.update(len(batch_data))
    pbar.close()
    f1_avg_score = sum(f1_score_list) / len(f1_score_list)
    print(f"f1-score: {f1_avg_score:.4f}")


# 使用vllm大模型加速推理框架对测试数据集进行评测
def test_with_vllm(model_path, test_path, batch_size: int = 8, quantization=None, kv_cache_dtype='auto'):
    test_dataset = load_json_data(test_path)
    f1_score_list = []
    pbar = tqdm(total=len(test_dataset), desc=f'progress')
    for i in range(0, len(test_dataset), batch_size):
        batch_data = test_dataset[i:i + batch_size]
        prompt_batch = [item["instruction"] + item["input"] for item in batch_data]
        # 调用vllm框架进行推理
        llm = ChatLLM(model_path, quantization=quantization, kv_cache_dtype=kv_cache_dtype)
        pred_label = llm.infer(prompt_batch)
        real_label = [item["output"] for item in batch_data]
        for pred_label, real_label in zip(pred_label, real_label):
            pred_label = pred_label.replace("'", '"')
            real_label = real_label.replace("'", '"')
            real_list = json.loads(real_label)
            try:
                pred_list = json.loads(pred_label)
                if len(pred_list) < len(real_list):
                    pred_list += ["O"] * (len(real_list) - len(pred_list))
                elif len(pred_list) > len(real_list):
                    pred_list = pred_list[:len(real_list)]
                f1 = f1_score(real_list, pred_list, average='micro')
                f1_score_list.append(f1)
            except json.JSONDecodeError:
                pass
        pbar.update(len(batch_data))
    pbar.close()
    f1_avg_score = sum(f1_score_list) / len(f1_score_list)
    print(f"f1-score: {f1_avg_score:.4f}")


# 对原始数据进行处理
def process_data(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 初始化变量
    dataset = []
    current_input = []
    current_output = []

    for line in lines:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 2:
                char, label = parts
                current_input.append(char)
                current_output.append(label)
            else:
                print(f"错误：数据格式不正确，跳过行：'{line}'")
        else:
            if current_input and current_output:
                dataset.append({
                    "instruction": "请对以下文本进行命名实体识别，输出每个字符的BIO标注。B表示实体的开始，I表示实体的内部，O表示非实体部分。最后以列表的格式输出结果：\n",
                    "input": str(current_input),
                    "output": str(current_output)
                })
                # 重置当前组数据
                current_input = []
                current_output = []

    # 处理最后一组数据（如果没有空行结尾）
    if current_input and current_output:
        dataset.append({
            "instruction": "请对以下文本进行命名实体识别，输出每个字符的BIO标注。B表示实体的开始，I表示实体的内部，O表示非实体部分。最后以列表的格式输出结果：\n",
            "input": str(current_input),
            "output": str(current_output)
        })

    # 保存数据
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, ensure_ascii=False, indent=2)

    print(f"处理完成，新的数据集已保存到 {output_file_path}")


if __name__ == '__main__':
    model_path = "../models/Qwen2-7B-Instruct"
    model, tokenizer = load_model(model_path, checkpoint_path='saved/medical_lora_model/checkpoint-1000')
    instruction = "请对以下文本进行命名实体识别，输出每个字符的BIO标注。B表示实体的开始，I表示实体的内部，O表示非实体部分。最后以列表的格式输出结果：\n"
    input = "['药', '进', '１', '０', '帖', '，', '黄', '疸', '稍', '退', '，', '饮', '食', '稍', '增', '，', '精', '神', '稍', '振']"
    prompt = instruction + input
    output = predict(model, tokenizer, prompt)
    print(output)
