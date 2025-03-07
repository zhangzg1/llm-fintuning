from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_dir = "../models/Qwen2-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
prompts = [
    "你好",
    "python是什么？",
    "为什么天空是蓝色的？",
]
batch_text = []
for q in prompts:
    text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    batch_text.append(text)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
llm = LLM(model=model_dir, trust_remote_code=True)
outputs = llm.generate(batch_text, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
