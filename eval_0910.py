from sklearn.metrics import roc_auc_score
import pandas as pd


# Modified function to include proper AUC calculation using roc_auc_score
def calculate_metrics_with_auc(csv_path, results_path):
    # Initialize counts for true positives, false positives, true negatives, and false negatives
    TP = FP = TN = FN = 0
    true_labels = []
    predicted_labels = []

    # Load CSV file
    df = pd.read_csv(csv_path)

    results = []

    # Iterate through each row in the DataFrame
    for i, row in df.iterrows():
        true_label = int(row[' True Label'])
        predicted_label = int(row[' Predicted Label']) if pd.notna(row[' Predicted Label']) else None

        if predicted_label is not None:
            # Update TP, FP, TN, FN based on true_label and predicted_label
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

            # Calculate metrics
            total = TP + TN + FP + FN
            accuracy = (TP + TN) / total if total else 0
            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            specificity = TN / (TN + FP) if (TN + FP) else 0

            # AUC calculation using sklearn's roc_auc_score
            auc = roc_auc_score(true_labels, predicted_labels) if len(set(true_labels)) > 1 else 0

            # Save the results
            result = {
                "Loan_ID": row['Loan_ID'],
                "True Label": true_label,
                "Predicted Label": predicted_label,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity
            }
            results.append(result)

            # Print data for inspection
            print(f"Loan Data {i}:\nTrue label: {true_label}\nPredicted label: {predicted_label}")

    # Create DataFrame from results and save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(results_path, index=False)
    print(f"Metrics saved to {results_path}")

# Example usage
calculate_metrics_with_auc("results/loan_prediction_metrics.csv", "loan_prediction_metrics.csv")
