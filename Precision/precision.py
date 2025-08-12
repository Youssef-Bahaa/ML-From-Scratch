def precision(y_true,y_pred,pos_label = 1):
    tp = fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yp == pos_label:
            if yt == yp:
                tp +=1
            else:
                fp +=1
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

y_true = [1, 0, 1, 1, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1]

print(precision(y_true , y_pred))




