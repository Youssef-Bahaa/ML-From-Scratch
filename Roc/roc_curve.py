import numpy as np
import matplotlib.pyplot as plt

def roc_curve(y_true,y_proba):
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    thresholds = list(np.unique(y_proba))
    thresholds.sort(reverse=True)
    thresholds.append(0)

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    TPR = []
    FPR = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))

        TPR.append(TP/P)
        FPR.append(FP/N)

    return FPR,TPR,thresholds


y_true = [0, 0, 1, 1, 1, 0, 1, 0, 0, 1,1]
y_scores = [0.05, 0.1, 0.4, 0.35, 0.8, 0.2, 0.7, 0.3, 0.15, 0.9,0.25]

FPR,TPR,threshold = roc_curve(y_true,y_scores)

plt.plot(FPR,TPR)
plt.show()
