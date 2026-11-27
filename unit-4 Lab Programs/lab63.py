# Lab 63 - Classification Precision Recall and F1 Score

actual =    ["A", "B", "A", "C", "B", "A", "C", "B"]
predicted = ["A", "B", "C", "C", "B", "A", "A", "B"]
classes = sorted(set(actual))
print("Complaint Classification Evaluation")
for label in classes:
    tp = sum(a == label and p == label for a, p in zip(actual, predicted))
    fp = sum(a != label and p == label for a, p in zip(actual, predicted))
    fn = sum(a == label and p != label for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    print("{}: Precision={:.2f}, Recall={:.2f}, F1={:.2f}".format(
        label, precision, recall, f1))
