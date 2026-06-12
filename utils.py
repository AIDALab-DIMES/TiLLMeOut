import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

def normalize_rescale(ts, bounds=[10, 110]):
    ts = (ts - ts.min()) / (ts.max() - ts.min()) * (bounds[1] - bounds[0]) + bounds[0]
    ts = ts.astype(int)
    return ts


def compute_AUC(y_true, y_pred, plot=False):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    if plot:
        plt.title(f"AUC = {roc_auc}")
        plt.plot(fpr, tpr)
        plt.show()
    return roc_auc


def compute_F1pa(y_true, errors):
    n = len(y_true)

    best_f1 = 0

    start = np.zeros(n, dtype=np.bool)
    stop = np.zeros(n, dtype=np.bool)

    start[0] = y_true[0] == 1
    start[1:n] = y_true[0:n - 1] < y_true[1:n]
    start = np.where(start)[0]
    stop[-1] = y_true[-1] == 1
    stop[0:n - 1] = y_true[0:n - 1] > y_true[1:n]
    stop = np.where(stop)[0] + 1

    for i, thres in enumerate(errors):

        tp = 0
        fn = 0
        fp = np.sum((1 - y_true)[errors >= thres])

        for j in range(len(start)):

            if np.sum(errors[start[j]:stop[j]] >= thres) > 0:
                tp += stop[j] - start[j]
            else:
                fn += stop[j] - start[j]

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * r * p) / (r + p)

        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def compute_delayed_F1(y_true, errors, delay):
    n = len(y_true)

    best_f1 = 0
    best_p = 0
    best_r = 0

    start = np.zeros(n, dtype=np.bool)
    stop = np.zeros(n, dtype=np.bool)

    start[0] = y_true[0] == 1
    start[1:n] = y_true[0:n - 1] < y_true[1:n]
    start = np.where(start)[0]
    stop[-1] = y_true[-1] == 1
    stop[0:n - 1] = y_true[0:n - 1] > y_true[1:n]
    stop = np.where(stop)[0] + 1

    for i, thres in enumerate(errors):

        tp = 0
        fn = 0
        fp = np.sum((1 - y_true)[errors >= thres])

        for j in range(len(start)):

            if np.sum(errors[start[j]:start[j] + delay] >= thres) > 0:
                tp += stop[j] - start[j]
            else:
                fn += stop[j] - start[j]

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = (2 * r * p) / (r + p)

        if f1 > best_f1:
            best_f1 = f1
            best_r = r
            best_p = p

    return best_p, best_r, best_f1


def compute_F1(y_true, y_pred):
    best_f1 = 0
    best_thres = 0

    for i, thres in enumerate(y_pred):

        f1 = f1_score(y_true, y_pred >= thres)
        if best_f1 < f1:
            best_f1 = f1
            best_thres = thres

    return best_thres, best_f1
