import numpy as np

class NumericalNB:
    def __init__(self):
        self.classes = []
        self.means = []
        self.variances = []
        self.priors = []

    def fit(self,x_train,y_train):
        self.classes = np.unique(y_train)

        for c in self.classes:
            x_c = x_train[y_train == c]

            prior = x_c.shape[0] / (y_train.shape[0])
            mean = np.mean(x_c,axis=0)
            var = np.var(x_c,axis=0)

            self.priors.append(prior)
            self.means.append(mean)
            self.variances.append(var)


    def gaussian(self,x,mean,variance):
        variance += 1e-9
        denomen = np.sqrt(2 * np.pi * variance)
        num = np.exp(-np.power(x - mean,2) / (2 * variance))
        return num / denomen

    def predict(self,X_test):
        y_preds = []
        for x_test in X_test:
            posteriors = []
            for idx,c in enumerate(self.classes):
                log_prior = np.log(self.priors[idx])
                log_likelihood = np.sum(np.log(self.gaussian(x_test,self.means[idx],self.variances[idx])))
                posterior = log_prior + log_likelihood
                posteriors.append(posterior)

            idx = np.argmax(posteriors)
            y_preds.append(self.classes[idx])

        return np.array(y_preds)

