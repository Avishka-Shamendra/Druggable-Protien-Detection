"""
Collection of functions which enable the evaluation of a classifier's performance,
by showing confusion matrix, accuracy, recall, precision etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn import metrics

def plot_confusion_matrix(conf_mat, label_strings=None, title='Confusion matrix', cmap=plt.cm.get_cmap('Blues')):
    """Plot confusion matrix in a separate window"""
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if label_strings:
        tick_marks = np.arange(len(label_strings))
        plt.xticks(tick_marks, label_strings, rotation=90)
        plt.yticks(tick_marks, label_strings)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def generate_classification_report(class_names, precision, recall, f1_score, support, confusion_matrix_row_normalized,
                                   digits=3, top_thieves=2, max_char_length=35):
    """
    Generates a classification report based on the input metrics.

    Args:
    class_names: A list of the class names.
    precision: An array of precision scores for each class.
    recall: An array of recall scores for each class.
    f1_score: An array of F1 scores for each class.
    support: An array of support values for each class.
    confusion_matrix_row_normalized: A matrix of normalized confusion values for each class.
    digits: Number of digits to be displayed after the decimal point.
    top_thieves: The number of top thieves to be displayed.
    max_char_length: The maximum number of characters to be used when displaying thief names.

    Returns:
    A string containing the classification report.
    """

    # Compute the relative frequency of each class in the true labels.
    rel_freq = support / np.sum(support)

    # Sort the class indices by importance (i.e. occurrence frequency).
    sorted_class_indices = np.argsort(rel_freq)[::-1]

    # Set up the report table.
    last_row_heading = 'avg / total'
    column_width = max(len(name) for name in class_names)
    column_width = max(column_width, len(last_row_heading), digits)
    headers = ["precision", "recall", "f1-score", "rel. freq.", "abs. freq.", "biggest thieves"]

    # Format the table.
    fmt = '%% %ds' % column_width
    fmt += '  '
    fmt += ' '.join(['% 10s' for _ in headers[:-1]])
    fmt += '|\t % 5s'
    fmt += '\n'

    # Add the headers to the report.
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'

    # Iterate through each class.
    for i in sorted_class_indices:
        # Add the class name to the values.
        values = [class_names[i]]

        # Add the precision, recall, F1 score, and relative frequency to the values.
        for v in (precision[i], recall[i], f1_score[i], rel_freq[i]):
            values += ["{0:0.{1}f}".format(v, digits)]

        # Add the absolute frequency to the values.
        values += ["{}".format(support[i])]

        # Find the top thieves for this class.
        thieves = np.argsort(confusion_matrix_row_normalized[i, :])[::-1][:top_thieves + 1]

        # Exclude the current class.
        thieves = thieves[thieves != i]

        # Compute the stealing ratio for each thief.
        steal_ratio = confusion_matrix_row_normalized[i, thieves]

        # Get the names of the thieves.
        thief_names = [
            class_names[thief][:min(max_char_length, len(class_names[thief]))] for thief in thieves
        ]

        # Combine the thief names and stealing ratios into a string.
        stealing_info = ""
        for j in range(len(thieves)):
            stealing_info += "{0}: {1:.3f},\t".format(thief_names[j], steal_ratio[j])

        # Add the stealing information to the values.
        values += [stealing_info]

        # Add the values to the report.
        report += fmt % tuple(values)

    report += '\n' + 100 * '-' + '\n'

    # Compute averages/sums.
    values = [last_row_heading]
    for v in (np.average(precision, weights=rel_freq), np.average(recall, weights=rel_freq),
                np.average(f1_score, weights=rel_freq)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(rel_freq))]
    values += ['{0}'.format(np.sum(support))]
    values += ['']

    # Add the total row to the report.
    report += fmt % tuple(values)

    return report


def evaluate_classification(y_pred, y_true, class_names, max_char_length=35, print_report=True,
                            show_plot=True, save_outputs=None):
    """
    For an array of label predictions and the respective true labels, shows confusion matrix, accuracy, recall,
    precision etc:

    Args:
    - y_pred: 1D array of predicted labels (class indices)
    - y_true: 1D array of true labels (class indices)
    - class_names: 1D array or list of class names in the order of class indices.
        Could also be integers [0, 1, ..., num_classes-1].
    - excluded_classes: list of classes to be excluded from average precision, recall calculation (e.g. OTHER)
    - max_char_length: maximum number of characters to show for each class name
    - print_report: whether to print classification report
    - show_plot: whether to show confusion matrix plot
    - save_outputs: folder path to save outputs (classification report, confusion matrix plot, metrics.json)

    Returns:
    - dictionary containing accuracy, precision, recall, F1-score, sensitivity, specificity

    """
    # Trim class_names to include only classes existing in y_pred OR y_true
    in_pred_labels = set(list(y_pred))
    in_true_labels = set(list(y_true))

    existing_class_indices = sorted(list(in_pred_labels | in_true_labels))
    class_strings = [str(name) for name in class_names]  # needed in case `class_names` elements are not strings
    existing_class_names = [class_strings[ind][:min(max_char_length, len(class_strings[ind]))] for ind in
                            existing_class_indices]  # a little inefficient but inconsequential

    # Confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    confusion_matrix_normalized_row = metrics.confusion_matrix(y_true, y_pred, normalize='true')

    # if show_plot:
    #     plt.figure()
    #     plot_confusion_matrix(confusion_matrix_normalized_row, label_strings=existing_class_names,
    #                            title='Confusion matrix normalized by row')
    #     plt.savefig(os.path.join(save_outputs, "confusion_matrix.png"), dpi=300)
    #     plt.show(block=False)

    # Analyze results
    total_accuracy = np.trace(confusion_matrix) / len(y_true)
    print('Overall accuracy: {:.3f}\n'.format(total_accuracy))

    # returns metrics for each class, in the same order as existing_class_names
    precision, recall, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                                             labels=existing_class_indices,
                                                                             zero_division=0)

    # Print classification report
    if print_report:
        print(generate_classification_report(existing_class_names, precision, recall, f1, support,
                                             confusion_matrix_normalized_row))

    output = {
        "accuracy": total_accuracy,
        "precision": precision.mean(),
        "recall": recall.mean(),
        "f1": f1.mean(),
        "sensitivity": recall[1],
        "specificity": recall[0]
    }

    if save_outputs:
        with open(os.path.join(save_outputs, "info_table.txt"), "w") as f:
            f.write(generate_classification_report(existing_class_names, precision, recall, f1, support,
                                                   confusion_matrix_normalized_row))

        with open(os.path.join(save_outputs, "metrics.json"), "w") as f:
            json.dump(output, f)

    return output
