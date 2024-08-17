import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Parameter Configuration')

    # loan data
    parser.add_argument('--loan_data', type=str, default='data/filtered_accepted_loans(FINAL).csv', help='path of loan data')

    # dataset
    parser.add_argument('--training_path', type=str, default='example/train/train.jsonl', help='path of training data')
    parser.add_argument('--val_path', type=str, default='example/val/val.jsonl', help='path of validation data')
    parser.add_argument('--test_path', type=str, default='example/test/test.jsonl', help='path of test data')

    # generative_model
    parser.add_argument('--generative_model', type=str, default='model/Mistral-7B-Instruct-v0.3', help='finetune model')

    # checkpoint
    parser.add_argument('--checkpoint', type=str, default='checkpoint/', help='path of checkpoint')

    # results
    parser.add_argument('--results', type=str, default='results/', help='the result path')

    return parser.parse_args()