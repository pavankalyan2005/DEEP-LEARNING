train_accuracy = 0.98
test_accuracy = 0.72
train_error = 1 - train_accuracy
test_error = 1 - test_accuracy
print("Training error:", train_error)
print("Testing error:", test_error)
print("Accuracy gap:", train_accuracy - test_accuracy)
if train_accuracy - test_accuracy > 0.15:
    print("The model is likely overfitting.")
    print("Use dropout, L2 regularization, early stopping, or more training data.")

