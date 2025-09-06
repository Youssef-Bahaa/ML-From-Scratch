import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

def train_test_split(X ,y ,test_size = 0.2 ,shuffle=True ,random_state = None):
    data_size = len(y)
    test_count = int(data_size * test_size)

    X = np.array(X)
    y = np.array(y)

    if shuffle:
        if random_state:
            np.random.seed(random_state)

        indices = np.arange(data_size)
        np.random.shuffle(indices)
        shuffled_x = []
        shuffled_y = []
        for i in indices:
            shuffled_x .append(X[i])
            shuffled_y .append(y[i])

    else:
        shuffled_x = X
        shuffled_y = y

    x_train = shuffled_x[: -test_count]
    x_test  = shuffled_x[-test_count:]

    y_train = shuffled_y[: -test_count]
    y_test  = shuffled_y[-test_count:]

    return x_train , x_test , y_train , y_test


diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

print(x_train[0])
print(x_test[0])






