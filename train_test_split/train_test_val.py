import numpy as np

def train_val_test(x,y,val_size,test_size,shuffle=True,random_state=42):
    x = np.array(x)
    y = np.array(y)

    size = len(x)
    val_count = int(val_size * size)
    test_count = int(test_size * size)
    train_count = size - (val_count + test_count)

    if shuffle:
        if random_state:
            np.random.seed(random_state)

        indices = np.arange(size)
        np.random.shuffle(indices)

        shuffle_x = x[indices]
        shuffle_y = y[indices]

    else:
        shuffle_x = x
        shuffle_y = y


    x_train = shuffle_x[:train_count]
    y_train = shuffle_y[:train_count]

    x_val = shuffle_x[train_count:train_count + val_count]
    y_val = shuffle_y[train_count:train_count + val_count]

    x_test = shuffle_x[train_count + val_count:]
    y_test = shuffle_y[train_count + val_count:]

    return x_train,x_val,x_test,y_train,y_val,y_test

