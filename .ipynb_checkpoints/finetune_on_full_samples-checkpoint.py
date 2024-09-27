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
from transformers import TrainingArguments


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

loan_data_path = "example"
dataset = load_dataset(loan_data_path)
train_data = dataset["train"]
train_data = [data for data in train_data]
random.shuffle(train_data)
dataset = Dataset.from_list(train_data)

print(f"Original dataset size: {len(dataset)}")

def rename_columns(example):
    example["loan_data"] = example.pop("text")
    example["labels"] = example.pop("label")
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
    use_gradient_checkpointing = True,
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
    outputs = examples["labels"]
    texts = []
    for inp, output in zip(inputs, outputs):
        # Add end of string token to prevent infinite generations.
        text = prompt.format(loan_data=inp, loan_status=output) + EOS_TOKEN
        texts.append(text)
    return {"text":texts}

# Building prompts
dataset= dataset.map(format_prompts, batched = True)

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
    output_dir = "outputs/mistral-7b-instruct-v0.3-0921",
    run_name = "mistral-7b-instruct-v0.3",
    logging_strategy = 'steps',
    logging_steps = 1,
    save_strategy="steps",
    save_steps=10,
    save_total_limit = 2,
    report_to = "wandb",
    )

# init the trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2, # if packing = False, else default to None
    packing = False, # more speed if packing short sequences. Maybe later
    args = training_args
)
print("trainer.dataset:", trainer.train_dataset)
print("trainer.train_dataset[0]:", trainer.train_dataset[0])

'''train'''
trainer = trainer.train()