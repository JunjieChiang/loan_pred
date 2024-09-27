import warnings
warnings.filterwarnings('ignore')
import os
import re
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from unsloth import FastLanguageModel
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTTrainer
from transformers import TrainingArguments, Trainer


# 贷款数据特征变量权重
feature_importance = {
    "Loan amount": 1.2,
    "DTI": 1.5,
    "Employment Title": 0.8,
    "Employment Length": 1.0,
    "Home Ownership": 1.1,
    "Annual Income": 1.6,
    "Verification Status": 1.0,
    "Grade": 2.0,
    "Purpose": 0.9,
    "Description": 0.7,
    "Title": 0.8,
    "Open Accounts": 1.3
}

# 将重要性列表转换为一个对应的tensor
importance_tensor = torch.tensor([v for v in feature_importance.values()], dtype=torch.float32)


# 定义自定义加权交叉熵损失函数
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, importance_tensor):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.importance_tensor = importance_tensor

    def forward(self, predictions, targets, features):
        # 确保 features 维度与 importance_tensor 匹配
        if features.shape[1] != len(self.importance_tensor):
            raise ValueError(
                f"Feature dimension {features.shape[1]} does not match importance tensor length {len(self.importance_tensor)}")

        # 确保 importance_tensor 和 features 在同一设备上
        self.importance_tensor = self.importance_tensor.to(features.device)

        # 如果 predictions 是 [batch_size, seq_len, num_classes]，需要展平为 [batch_size * seq_len, num_classes]
        if predictions.dim() == 3:
            batch_size, seq_len, num_classes = predictions.size()
            predictions = predictions.view(-1, num_classes)  # 展平为 [batch_size * seq_len, num_classes]

        # 如果 targets 是 [batch_size, seq_len]，需要展平为 [batch_size * seq_len]
        if targets.dim() == 1:
            # 假设每个样本在序列中有相同的标签，重复 targets 以匹配序列长度
            targets = targets.unsqueeze(1).expand(-1, seq_len).contiguous()

        # 确保 targets 被展平为 [batch_size * seq_len]
        targets = targets.view(-1)

        # 确保 predictions 和 targets 的 batch_size 一致
        if predictions.size(0) != targets.size(0):
            raise ValueError(
                f"Expected predictions and targets to have the same batch_size after flattening, but got {predictions.size(0)} and {targets.size(0)}")

        # 计算标准交叉熵损失
        cross_entropy_loss = nn.CrossEntropyLoss()(predictions, targets)

        # 将特征张量展平并乘以权重
        weighted_features = features.view(features.size(0), -1) * self.importance_tensor.unsqueeze(0)

        # 使用加权特征对整体损失进行加权
        weighted_loss = cross_entropy_loss * torch.mean(weighted_features)

        return weighted_loss

loan_data_path = "example"
dataset = load_dataset(loan_data_path)

train_data = dataset["train"]

# balance data sample
label_1_data = [data for data in train_data if data['label'] == 1]
label_0_data = [data for data in train_data if data['label'] == 0]

num_label_1 = len(label_1_data)
balanced_label_0_data = random.sample(label_0_data, num_label_1)
balanced_data = label_1_data + balanced_label_0_data

# random data layout
random.shuffle(balanced_data)

dataset = Dataset.from_list(balanced_data)

def rename_columns(example):
    example["loan_data"] = example.pop("text")
    example["label"] = example.pop("label")
    return example

dataset = dataset.map(rename_columns)


max_seq_length = 2048
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # True to use 4bit quantization / reduce memory usage (for T4 GPU)

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/mistral-7b-v0.2-bnb-4bit",
    model_name = "model/Mistral-7B-Instruct-v0.3",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # can improve fine-tuning, at attention/feed fwd layers
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 16, # more change to pre-train weights but care overfitting
    lora_dropout = 0.05, # any, but 0 if perf opti.
    bias = "none",    # any, but "none" is perf  opti.
    random_state = 11,
    use_rslora = False,  # support rank stabilized LoRA
    loftq_config = None, # LoftQ support
)

# Prompt Preparation
prompt = """You are a highly intelligent and detailed artificial intelligence assistant with a deep understanding of financial data, specifically in predicting loan defaults.
Your task is to accurately classify loan data into one of two possible outcomes:
- 0: The loan is fully paid off (no default).
- 1: The loan has defaulted (borrower failed to meet the repayment terms).

The input data will provide various details about the loan and the borrower's financial situation. Your goal is to carefully analyze this information and determine the loan's status based on the provided features.

You are expected to generate a response that is one of the following labels:
- 0: The loan is fully paid off.
- 1: The loan has defaulted.

Your classification must be precise and match the best possible outcome for the given loan data.

Here is the loan data you need to classify:
### Loan Data:
{loan_data}
### Loan Status:
{loan_status}"""

# Add EOS special token, according to previously loaded tokenizer
EOS_TOKEN = tokenizer.eos_token
def format_prompts(examples):
    inputs = examples["loan_data"]
    outputs = examples["label"]
    texts = []
    for inp, output in zip(inputs, outputs):
        # Add end of string token to prevent infinite generations.
        text = prompt.format(loan_data=inp, loan_status=output) + EOS_TOKEN
        texts.append(text)
    return {"text":texts}

# Building prompts
dataset= dataset.map(format_prompts, batched = True)


# 自定义Trainer以使用自定义损失函数
class CustomTrainer(Trainer):
    def __init__(self, *args, dataset_text_field=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_text_field = dataset_text_field  # 添加 dataset_text_field 字段
        self.train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)

        self.train_dataset = self.train_dataset.remove_columns(["loan_data", "text"])

    def tokenize_function(self, examples):
        # 使用 tokenizer 对 text 进行编码，生成 input_ids 和 attention_mask
        return self.tokenizer(examples[self.dataset_text_field], padding="max_length", truncation=True)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 如果有自定义的 dataset_text_field 使用它
        # print("inputs:", inputs)
        inputs = {key: val.to(model.device) for key, val in inputs.items()}  # 将 inputs 移动到模型所在的设备

        # 获取模型的输出
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        logits = outputs.logits
        labels = inputs["labels"]

        # 获取特征（假设特征在 inputs["features"]）
        features = inputs["features"] if "features" in inputs else torch.ones(
            (logits.shape[0], len(feature_importance))).to(logits.device)

        # 调用自定义损失函数
        loss = WeightedCrossEntropyLoss(importance_tensor)(logits, labels, features)

        return (loss, outputs) if return_outputs else loss

# Train the model
training_args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    num_train_epochs = 1,
    # max_steps = 110,
    learning_rate = 2e-4, # 2e-5
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 11,
    output_dir = "outputs/mistral-7b-instruct-v0.3-0910",
    run_name = "mistral-7b-instruct-v0.3",
    logging_strategy = 'steps',
    logging_steps = 1,
    save_strategy="steps",
    save_steps=10,
    save_total_limit = 2,
    report_to = "wandb",
    )

# init the trainer
trainer = CustomTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    args = training_args
)
print("trainer.dataset:", trainer.train_dataset)


'''train'''
trainer = trainer.train()