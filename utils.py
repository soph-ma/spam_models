import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for true_label, predicted_label in zip(y_true, y_pred):
        if true_label == 1 and predicted_label == 1:
            true_positive += 1
        elif true_label == 0 and predicted_label == 0:
            true_negative += 1
        elif true_label == 0 and predicted_label == 1:
            false_positive += 1
        elif true_label == 1 and predicted_label == 0:
            false_negative += 1

    confusion_matrix = np.array([[true_negative, false_positive], [false_negative, true_positive]])
    return confusion_matrix