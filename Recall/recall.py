
def recall(y_true,y_pred,pos_label = 1):
    tp = fn = 0
    for yt,yp in zip(y_true,y_pred):
       if yt == yp  == pos_label:
           tp += 1
       elif yp != pos_label and yt == pos_label:
           fn += 1
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]

print(recall(y_true , y_pred))




