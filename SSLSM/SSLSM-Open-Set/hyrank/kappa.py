import numpy as np

def kappa(testData, k=None):
    """
    Compute Cohen's Kappa and Average Accuracy (AA).
    Fully compatible with NumPy 2.0+.
    """
    dataMat = np.asarray(testData, dtype=np.float64)
    s = np.sum(dataMat)
    if s == 0:
        return 0.0, 0.0, np.zeros(dataMat.shape[0])
    
    P0 = np.trace(dataMat) / float(s)
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    Pe = float(np.sum(xsum * ysum)) / float(s ** 2)
    cohens_coefficient = float((P0 - Pe) / (1.0 - Pe)) if (1.0 - Pe) != 0 else 0.0

    if k is None:
        k = dataMat.shape[0]

    col_sum = dataMat.sum(axis=0)
    a = np.zeros(k, dtype=np.float64)
    for i in range(k):
        if col_sum[i] != 0:
            a[i] = float(dataMat[i, i]) / float(col_sum[i])
        else:
            a[i] = 0.0

    return cohens_coefficient, a.mean() * 100, a * 100
