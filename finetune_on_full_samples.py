import warnings
warnings.filterwarnings('ignore')
import os
import random
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


# 加载数据集
loan_data_path = "example"
dataset = load_dataset(loan_data_path)

# 数据预处理
def preprocess_data(examples):
    examples["loan_data"] = examples.pop("text")
    examples["labels"] = int(examples.pop("label"))
    return examples

# 预处理训练集
train_data = dataset["train"]
train_data = train_data.map(preprocess_data)
train_data = [data for data in train_data]


# 计算类别权重
def compute_class_weights(data):
    labels = [example['labels'] for example in data]
    label_counts = Counter(labels)
    total_samples = sum(label_counts.values())
    class_weights = [total_samples / label_counts[i] for i in range(2)]
    return torch.tensor(class_weights, dtype=torch.float)

class_weights = compute_class_weights(train_data)


# 过采样少数类
def oversample_data(data):
    labels = [example['labels'] for example in data]
    label_counts = Counter(labels)
    majority_class = label_counts.most_common(1)[0][0]
    minority_class = 1 - majority_class
    minority_data = [d for d in data if d['labels'] == minority_class]
    oversampled_minority_data = minority_data * (label_counts[majority_class] // label_counts[minority_class])
    return data + oversampled_minority_data


balanced_train_data = oversample_data(train_data)
random.shuffle(balanced_train_data)
dataset = Dataset.from_list(balanced_train_data)

# 初始化模型和分词器
model_name = "model/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 数据集预处理函数
def tokenize_function(examples):
    return tokenizer(
        examples["loan_data"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 设置格式
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 创建加权随机采样器
def create_weighted_sampler(data):
    labels = data['labels']
    class_sample_count = np.array([len(np.where(labels.numpy() == t)[0]) for t in np.unique(labels.numpy())])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels.numpy()])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

sampler = create_weighted_sampler(tokenized_datasets)


# 自定义损失函数
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# 训练参数
training_args = TrainingArguments(
    output_dir="outputs/mistral-7b-instruct-v0.3-0922",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    evaluation_strategy="no",
    report_to="wandb",
    fp16=torch.cuda.is_available(),
)

# 创建数据加载器
train_dataloader = DataLoader(
    tokenized_datasets,
    sampler=sampler,
    batch_size=training_args.per_device_train_batch_size,
)

# 初始化 Trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=None,
)

# 开始训练
trainer.train()