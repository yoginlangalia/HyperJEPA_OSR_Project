#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cohen's Kappa and Average Accuracy (AA) metric calculations.
Compatible with NumPy 1.x and NumPy 2.0+.
"""

import numpy as np

def kappa(confusion_matrix, k):
    """
    Calculate Cohen's Kappa, Average Accuracy (AA in %), and per-class accuracies (in %).
    Uses pure ndarray math compatible with NumPy 1.x and NumPy 2.0+.
    """
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    total = np.sum(cm)
    if total == 0:
        return 0.0, 0.0, np.zeros(k)

    P0 = np.trace(cm) / total
    xsum = np.sum(cm, axis=1)  # row sums (true class counts)
    ysum = np.sum(cm, axis=0)  # col sums (predicted class counts)

    Pe = np.sum(xsum * ysum) / (total ** 2)

    if (1.0 - Pe) == 0:
        cohen_kappa = 1.0
    else:
        cohen_kappa = (P0 - Pe) / (1.0 - Pe)

    each_acc = np.zeros(k)
    for i in range(k):
        if xsum[i] > 0:
            each_acc[i] = cm[i, i] / xsum[i]
        else:
            each_acc[i] = 0.0

    aa = np.mean(each_acc) * 100.0
    return float(cohen_kappa), float(aa), each_acc * 100.0
