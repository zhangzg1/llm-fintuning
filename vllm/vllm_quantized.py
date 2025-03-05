from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/root/llm_lora/models/Qwen2_7B_Instruct_lora'
quant_path = '/root/llm_lora/models/Qwen2_7B_Instruct_lora_awq'
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True, "use_cache": False})
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化基础模型，如果无法从huggingface下载数据集，可以load到本地然后加载
dataset_path = "/root/llm_lora/dataset/mit-han-lab/pile-val-backup"
model.quantize(tokenizer, quant_config=quant_config, calib_data=dataset_path, split="validation")
# 量化基础模型，直接连接huggingface并加载相关数据集，
# model.quantize(tokenizer, quant_config=quant_config)

# 保存量化模型
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
