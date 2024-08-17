import pandas as pd
from config import get_args
from datasets import Dataset
from sklearn.model_selection import train_test_split


args = get_args()

data = pd.read_csv(args.loan_data)

'''选择重要的特征变量'''
selected_features = ['loan_amnt', 'dti', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc',
                     'verification_status', 'grade', 'sub_grade', 'purpose', 'desc', 'title', 'open_acc', 'loan_status']

data = data[selected_features]

'''将目标变量 loan_status 转换为二分类, 逾期是 Default, 值为 1, 不逾期是 Fully Paid, 值为 0'''
data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x == 'Default' else 0)

'''合并特征为文本格式'''
data['text'] = data.apply(lambda x: f"Loan amount: {x['loan_amnt']}, DTI: {x['dti']}, Employment Title: {x['emp_title']}, Employment Length: {x['emp_length']}, Home Ownership: {x['home_ownership']}, Annual Income: {x['annual_inc']}, Verification Status: {x['verification_status']}, Grade: {x['grade']}-{x['sub_grade']}, Purpose: {x['purpose']}, Description: {x['desc']}, Title: {x['title']}, Open Accounts: {x['open_acc']}", axis=1)

'''切分数据集'''
train_texts, val_texts, train_labels, val_labels = train_test_split(data['text'], data['loan_status'], test_size=0.2)

'''创建 Hugging Face 数据集'''
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
train_dataset.to_json(args.training_path, orient="records", lines=True)
val_dataset.to_json(args.val_path, orient="records", lines=True)