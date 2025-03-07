from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch


# 释放gpu显存
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device("cuda:0"):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatLLM(object):
    def __init__(self, model_path, quantization=None, kv_cache_dtype='auto'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        # 加载 vLLM 框架的大模型
        self.model = LLM(model=model_path,
                         tokenizer=model_path,
                         tensor_parallel_size=1,          # 如果是多卡，可以自己把这个并行度设置为卡数N
                         quantization=quantization,       # 是否加载量化模型
                         kv_cache_dtype=kv_cache_dtype,   # 是否使用KV缓存
                         gpu_memory_utilization=0.9,      # 可以根据gpu的利用率自己调整这个比例，默认0.9
                         trust_remote_code=True,
                         dtype="bfloat16")
        # LLM的采样参数，这里有些参数是使用大模型本身默认的，有些是自己设置的
        sampling_kwargs = {
            "stop_token_ids": [self.tokenizer.eos_token_id],
            "early_stopping": False,
            "top_p": 1.0,
            "top_k": -1,                   # 当使用束搜索时top_k必须为-1
            "temperature": 0.0,
            "max_tokens": 2000,
            "repetition_penalty": 1.05,
            "n": 1,
            "best_of": 2,                  # 生成的候选数量和最佳选择数量
            "use_beam_search": True        # 是否使用束搜索
        }
        self.sampling_params = SamplingParams(**sampling_kwargs)

    # 批量推理，输入一个batch，返回一个batch的答案
    def infer(self, prompts):
        batch_text = []
        for q in prompts:
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
            batch_text.append(text)
        # 调用vllm框架和大模型来批量生成答案
        outputs = self.model.generate(batch_text, sampling_params=self.sampling_params)
        batch_response = []
        for output in outputs:
            output_str = output.outputs[0].text
            # 如果结束符token在文本中，则去除该标志后面的文本
            if self.tokenizer.eos_token in output_str:
                output_str = output_str[:-len(self.tokenizer.eos_token)]
            # 如果填充符token在文本中，则去除该标志后面的文本
            if self.tokenizer.pad_token in output_str:
                output_str = output_str[:-len(self.tokenizer.pad_token)]
            batch_response.append(output_str)
        torch_gc()
        return batch_response
