import numpy as np

def confusion_matrix(y_true, y_pred):
    tn = fp = fn = tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            if yt == 0:
                tn += 1
            else:
                tp += 1
        else:
            if yp == 0:
                fn += 1
            else:
                fp += 1
    return np.array([[tn, fp],
           [fn, tp]])

y_pred = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_true =  [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]

tn , fp , fn , tp = confusion_matrix(y_true, y_pred).ravel()
print(f'{tn}  {fp}')
print(f'{fn}  {tp}')