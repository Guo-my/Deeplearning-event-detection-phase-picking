import numpy as np
import torch.nn as nn
import torch


def Event_F1(y_true, y_pred):
    """

    Calculate F1-score.

    Parameters
    ----------
    y_true : 1D array
        Ground truth labels.

    y_pred : 1D array
        Predicted labels.

    Returns
    -------
    f1 : float
        Calculated F1-score.

    """

    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        if all(x == 0 for x in y_true):
            indices = np.where(y_pred > 0.5)[0]
            if len(indices) == 0:
                re = 1
            else:
                re = 0
        else:
            expanded_arr = np.copy(y_pred)
            expanded_arr = np.where(expanded_arr > 0.5, 1, expanded_arr)

            true_positives = np.sum(np.round(np.clip(y_true * expanded_arr, 0, 1)))
            possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
            re = true_positives / (possible_positives + 1e-8)
        return re

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        if all(x == 0 for x in y_true):
            indices = np.where(y_pred > 0.5)[0]
            if len(indices) == 0:
                pre = 1
            else:
                pre = 0
        else:
            expanded_arr = np.copy(y_pred)
            expanded_arr = np.where(expanded_arr > 0.5, 1, expanded_arr)

            expanded_true = np.copy(y_true)
            expanded_true = np.where(expanded_true > 0.5, 1, expanded_true)

            true_positives = np.sum(np.round(np.clip(expanded_arr * expanded_true, 0, 1)))
            predicted_positives = np.sum(np.round(np.clip(expanded_arr, 0, 1)))
            pre = true_positives / (predicted_positives + 1e-8)
        return pre

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return recall, precision, 2 * ((precision * recall) / (precision + recall + 1e-8))


def recall(ground_true, pred):
    'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

    Tp = 0
    Fp = 0
    Fn = 0
    for i in range(ground_true.shape[0]):
        y_true = ground_true[i]
        y_pred = pred[i, 0, :]
        if all(x == 0 for x in y_true):
            indices = np.where(y_pred > 0.3)[0]
            if len(indices) == 0:
                Tp += 1
            else:
                Fp += 1
        else:
            true_indices = int(np.where(y_true == 1)[0])
            indices = np.where(y_pred >= 0.3)[0]
            if len(indices) == 0:
                Fn += 1
            else:
                res = np.abs(indices - true_indices)
                if min(res) <= 50:
                    Tp += 1
                else:
                    Fn += 1

    pr = Tp / (Tp + Fp)
    re = Tp / (Tp + Fn)
    f1 = (2*(pr * re))/(pr + re)

    return re, pr, f1
