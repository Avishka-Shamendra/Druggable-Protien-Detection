"""
Util funtions to get accuracy,  sensitivity, specificity, precision, F1-score
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn import metrics


def evaluate_classification(y_pred, y_true, class_names, save_outputs=None):
    """
    Args:
    - y_pred: 1D array of predicted labels (class indices)
    - y_true: 1D array of true labels (class indices)
    - class_names: 1D array or list of class names in the order of class indices.
        Could also be integers [0, 1, ..., num_classes-1].
    - save_outputs: folder path to save outputs

    Returns:
    - dictionary containing accuracy,  sensitivity, specificity, precision, F1-score

    """
    # Trim class_names to include only classes existing in y_pred OR y_true
    in_pred_labels = set(list(y_pred))
    in_true_labels = set(list(y_true))

    existing_class_indices = sorted(list(in_pred_labels | in_true_labels))
    class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    confusion_matrix_normalized_row = metrics.confusion_matrix(y_true, y_pred, normalize='true')

    # Analyze results
    total_accuracy = np.trace(confusion_matrix) / len(y_true)

    # returns metrics for each class, in the same order as existing_class_names
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                             labels=existing_class_indices,
                                                                             zero_division=0)
    output = {
        "accuracy": total_accuracy,
        "sensitivity": recall[1],
        "specificity": recall[0],
        "precision": precision.mean(),
        "f1": f1.mean()
    }

    if save_outputs:
        with open(os.path.join(save_outputs, "report.json"), "w") as f:
            json.dump(output, f)

    return output
