import numpy as np

class CategoricalNB:
    def __init__(self):
        self.priors = []
        self.classes = []
        self.likelihood = {}
    def fit(self,x_train,y_train):
        self.classes = np.unique(y_train)
        for c in self.classes:
            filtered = x_train[y_train == c]
            prior = filtered.shape[0] / y_train.shape[0]
            self.priors.append(prior)

            self.likelihood[c] = {}
            for col in range(x_train.shape[1]):
                self.likelihood[c][col] = {}

                values, counts = np.unique(filtered[: , col] , return_counts = True)
                total = counts.sum()
                for idx , cnt in enumerate(counts):
                     self.likelihood[c][col][values[idx]] = (1 + cnt) / (len(values)+ filtered.shape[0])


    def predict(self,X_test):
        y_preds = []
        for x_test in X_test:
            classes_max = []
            for idx , c in enumerate(self.classes):
                posterior_log = np.log(self.priors[idx])
                for col in range(X_test.shape[1]):
                    val = x_test[col]
                    if val in self.likelihood[c][col]:
                        posterior_log += np.log(self.likelihood[c][col][val])
                    else:
                        unseen_prob = 1 / (sum(self.likelihood[c][col].values()) + len(self.likelihood[c][col]))
                        posterior_log += np.log(unseen_prob)

                classes_max.append(posterior_log)

            idx = np.argmax(classes_max)
            y_preds.append(self.classes[idx])

        return np.array(y_preds)


