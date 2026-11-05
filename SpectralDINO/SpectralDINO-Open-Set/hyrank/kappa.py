#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cohen's Kappa and Average Accuracy (AA) metric calculations."""

import numpy as np

def kappa(confusion_matrix, k):
    dataMat = np.mat(confusion_matrix)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(ysum * xsum)
    P0 = float(P0 / (float(xsum.sum())))
    Pe = Pe / (float(xsum.sum()) ** 2)
    oa = P0
    cohen_kappa = (P0 - Pe) / (1 - Pe)
    each_acc = np.zeros(k)
    for i in range(k):
        if xsum[i, 0] > 0:
            each_acc[i] = dataMat[i, i] / xsum[i, 0]
        else:
            each_acc[i] = 0.0
    aa = np.mean(each_acc)
    return cohen_kappa, aa, each_acc
