# 实现peft框架和llama-factory框架Lora微调Qwen2，Putning微调chatglm3-6b，完成中医药命名实体识别任务，并使用vllm框架加速推理

## 1、介绍

该项目主要是利用大模型微调完成中医药命名实体识别（NER）的任务，最后基于大模型推理加速框架 vllm 进行批次评测。这里的数据集主要是中医药领域相关的数据集，可用于命名实体识别等自然语言处理任务，数据经过处理后划分为训练集、验证集和测试集共约 6000 条标注数据。命名实体识别可以作为一项独立的信息抽取任务，在许多语言处理技术大型应用系统中扮演了关键的角色，如信息检索、自动文本摘要、问答系统、机器翻译等。

这里我们选择以 qwen2-7B 和 chatglm3-6b 为基座大模型，然后分别使用 peft 框架和 llama-factory 框架对 qwen2 模型进行 Lora 微调，使用 Ptuning 方法微调 chatglm3-6b 模型，总共实现了三种微调方法来完成中医药命名实体识别任务。最后基于 vllm 框架使用测试数据集对模型进行评测，模型的评价指标采用词的 F1 score，不是单字的。

## 2、显卡要求

24G 显存及以上（4090或V100及以上），一块或多块。

## 3、下载源码

```
git clone https://github.com/zhangzg1/llm-fintuning.git
cd llm-fintuning
```

## 4、安装依赖环境（Linux）

```
# 创建虚拟环境
conda create -n llm-fintuning python=3.10
conda activate llm-fintuning
# 安装其他依赖包
pip install -r requirements.txt
```

## 5、代码结构

```text
.
├── dataset                           
    ├── ChineseMedical
        └── medical_train        # 原始训练数据
        └── medical_eval         # 原始评估数据
        └── medical_test         # 原始测试数据
    ├── mit-han-lab              # AWQ量化模型所使用的数据集
├── llama_factory
├── models
    ├── Qwen2-7B-Instruct        # 基座大模型
    ├── chatglm3-6b
├── peft_code
    ├── merge_model_lora.py           # 合并lora导出模型
    ├── peft_lora.py                  # lora微调配置
    ├── peft_util.py                  # 微调相关函数
├── ptuning
├── saved                        # 存储lora微调的模型
├── vllm_code
    ├── model.test.py            # 测试vllm框架
    ├── vllm_predict.py          # vllm框架批次预测
    ├── vllm_quantized.py        # 模型量化
├── run.py                       # 主函数
├── requirements.txt             # 第三方依赖库
├── README.md                    # 说明文档             
```

## 6、微调与测试

```
# 方法一：基于peft框架对qwne2模型进行lora微调，并在微调后进行评测
python run.py

# 方法二：基于llama-factory框架对qwne2模型进行lora微调
cd ptuning/
bash train.sh         # 微调模型
bash export_qwen.sh   # 导出模型

# 方法三：使用Ptuning方法微调chatglm3模型
cd llama_factory/
bash train.sh         # 微调模型
bash evaluate.sh      # 评估模型

```

这三种微调方法都可以利用数据集微调大模型来完成中医药命名实体识别（NER）的任务。在使用 Ptuning 方法微调 chatglm3 模型时，由于官方的模型文件里面可能有错误，详细解决方法可以看[models/chatglm3-6b/README.md](https://github.com/zhangzg1/llm-fintuning/main/models/chatglm3-6b/README.md)