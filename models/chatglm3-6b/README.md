Hugging Face download URL：https://huggingface.co/THUDM/chatglm3-6b

Modelscope download URL：https://www.modelscope.cn/models/ZhipuAI/chatglm3-6b

## TypeError: _pad() got an unexpected keyword argument 'padding_side'

Ptuning微调时出现该错误时，需要在修改模型中的 tokenization_chatglm.py 文件（**transformers >=4.45.0**）：

```
def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            添加代码：padding_side: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
```

## AttributeError: 'ChatGLMTokenizer' object has no attribute 'build_prompt'

Ptuning微调时出现该错误时，需要在修改模型中的 tokenization_chatglm.py 文件，在类'ChatGLMTokenizer'中添加函数 'build_prompt'：

```
def build_prompt(self, query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt
```

在chatglm2-6b中包含这个函数，如果Ptuning微调chatglm2可以不用添加。