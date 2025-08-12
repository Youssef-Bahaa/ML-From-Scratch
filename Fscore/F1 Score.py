def precision(y_true,y_pred,pos_label = 1):
    tp = fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yp == pos_label:
            if yt == yp:
                tp +=1
            else:
                fp +=1
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true,y_pred,pos_label = 1):
    tp = fn = 0
    for yt,yp in zip(y_true,y_pred):
       if yt == yp  == pos_label:
           tp += 1
       elif yp != pos_label and yt == pos_label:
           fn += 1
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(p,r):
    return (2 * p * r) / (p + r)




y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]

p = precision(y_true, y_pred)
r = recall(y_true, y_pred)
f1 = f1_score(p, r)

print("Precision:", p)
print("Recall:", r)
print("F1 Score:", f1)
