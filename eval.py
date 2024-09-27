import torch
import re
import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="outputs/mistral-7b-instruct-v0.3-0910/checkpoint-240",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


loan_data_path = "example"
dataset = load_dataset(loan_data_path)
val_data = dataset["validation"]

label_1_data = [data for data in val_data if data['label'] == 1]
label_0_data = [data for data in val_data if data['label'] == 0]

num_label_1 = len(label_1_data)
balanced_label_0_data = random.sample(label_0_data, num_label_1)
balanced_data = label_1_data + balanced_label_0_data

# random data layout
random.shuffle(balanced_data)

dataset = Dataset.from_list(balanced_data)


def rename_columns(data):
    data["loan_data"] = data.pop("text")
    data["loan_status"] = data.pop("label")
    return data


dataset = dataset.map(rename_columns)

prompt = """You are a highly intelligent and detailed artificial intelligence assistant with a deep understanding of financial data, specifically in predicting loan defaults.
Your task is to predict the given loan data into one of two possible outcomes:
- 0: The loan is fully paid off (no default).
- 1: The loan has defaulted (borrower failed to meet the repayment terms).

Here is the loan data you need to classify:
### Loan Data:
{}

### Predicted Loan Status:
"""

results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_path = os.path.join(results_dir, "loan_prediction_metrics.csv")

if not os.path.exists(results_path):
    with open(results_path, 'w') as f:
        f.write("Loan_ID, True Label, Predicted Label, Accuracy, F1-Score, AUC, Precision, Recall, Specificity\n")

TP = 0
FP = 0
TN = 0
FN = 0

true_labels = []
predicted_labels = []

for i, data in enumerate(tqdm(dataset, desc="Evaluating")):
    inputs = tokenizer(
        [
            prompt.format(
                data['loan_data'],
                "",
            )
        ],
        return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=20, use_cache=True)
    decoded_outputs = tokenizer.batch_decode(outputs)

    matches = re.findall(r'Predicted Loan Status:\n(\d)', decoded_outputs[0])
    predicted_label = int(matches[-1]) if matches else None

    if predicted_label is not None:
        true_label = int(data['loan_status'])

        if true_label == 1 and predicted_label == 1:
            TP += 1
        elif true_label == 0 and predicted_label == 1:
            FP += 1
        elif true_label == 0 and predicted_label == 0:
            TN += 1
        elif true_label == 1 and predicted_label == 0:
            FN += 1

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        specificity = TN / (TN + FP) if (TN + FP) else 0

        TPR = recall
        FPR = FP / (FP + TN) if (FP + TN) else 0
        auc = (TPR * (1 - FPR) + (1 - TPR) * FPR) / 2

        result = {
            "Loan_ID": i,
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "Specificity": specificity
        }

        result_df = pd.DataFrame([result])
        result_df.to_csv(results_path, mode='a', header=False, index=False)

        print(f"Loan Data {i}:\n{data['loan_data']}\nTrue label: {true_label}\nPredicted label:\n{predicted_label}")

print(f"Metrics saved to {results_path}")