# Lora微调Qwen2完成中医药命名实体识别任务，并使用vllm框架加速推理

## 1、介绍
 
该项目主要是利用大模型微调完成中医药命名实体识别（NER）的任务，最后基于大模型推理加速框架vllm进行批次评测。这里的数据集主要是中医药领域相关的数据集，可用于命名实体识别等自然语言处理任务，数据经过处理后划分为训练集、验证集和测试集共约6000条标注数据。命名实体识别可以作为一项独立的信息抽取任务，在许多语言处理技术大型应用系统中扮演了关键的角色，如信息检索、自动文本摘要、问答系统、机器翻译等。

这里我们选择qwen2-7B为基座大模型，然后基于peft框架对基座模型进行Lora微调，最后基于vllm框架使用测试数据集对模型进行评测，模型的评价指标采用词的 F1 score，不是单字的。

## 2、显卡要求

24G显存及以上（4090或V100及以上），一块或多块。

## 3、下载源码

```
git clone https://github.com/zhangzg1/llm_lora.git
cd llm_lora
```

## 4、安装依赖缓解（Linux）

```
# 创建虚拟环境
conda create -n llm_lora python=3.10
conda activate llm_lora
# 安装其他依赖包
pip install -r requirements.txt
```

## 5、代码结构

```text
.
├── dataset                           
    ├── ChineseMedical
		└── medical_train      # 原始训练数据
		└── medical_eval       # 原始评估数据
		└── medical_test       # 原始测试数据
    ├── mit-han-lab            # AWQ量化模型所使用的数据集
├── models
	├── Qwen2_7B_Instruct      # 基座大模型
├── saves                      # 存储lora微调的模型
├── vllm
	├── model.test.py          # 测试vllm框架
	├── vllm_predict.py        # vllm框架批次预测
	├── vllm_quantized.py      # 模型量化
├── mian.py                    # 主函数
├── merge_model_lora.py        # 合并lora导出模型
├── NER_lora.py                # lora微调配置
├── NER_util.py                # 微调相关函数
├── requirements.txt           # 第三方依赖库
├── README.md                  # 说明文档             
```

## 6、微调与测试

```
# 运行主函数，基座模型进行lora微调，并在微调后进行评测
python main.py
```

