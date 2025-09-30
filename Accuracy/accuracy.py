
def accuracy_score(y_true,y_pred):
    n = len(y_pred)
    true = 0
    for i in range(n):
        if y_pred[i] == y_true[i]:
            true += 1

    return true / n


y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]

accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

