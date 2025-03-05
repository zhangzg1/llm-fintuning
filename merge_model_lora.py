from transformers import AutoModelForCausalLM, LlamaTokenizer
from peft import PeftModel
import torch


def save_lora_model(base_model_path, save_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)
    base_tokenizer = LlamaTokenizer.from_pretrained(base_model_path)

    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {save_path}")
    model.save_pretrained(save_path)
    base_tokenizer.save_pretrained(save_path)


if __name__ == "__main__":
    base_model_path = "/root/llm_lora/models/Qwen2_7B_Instruct"
    save_path = "/root/llm_lora/models/Qwen2_7B_Instruct_lora"
    lora_path = "/root/llm_lora/saves/medical_lora_model/checkpoint-1000"
    save_lora_model(base_model_path, save_path, lora_path)
