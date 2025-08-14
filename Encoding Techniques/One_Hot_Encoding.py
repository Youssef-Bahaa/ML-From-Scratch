import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.columns = []

    def fit(self,labels):
        self.columns = sorted(set(labels))

    def transform(self,labels):
        encoded_data = []
        for val in labels:
            row = [1 if val == col else 0 for col in self.columns]
            encoded_data.append(row)

        return np.array(encoded_data)

    def fit_transform(self,labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self,encoding):
        output = []
        for idx in range(len(encoding)):
            for col,val in enumerate(encoding[idx]):
                if val == 1:
                    output.append(self.columns[col])

        return output

encoder = OneHotEncoder()
labels = ["cat", "dog", "cat", "bird"]
encoded = encoder.fit_transform(labels)
print("Encoded:\n", encoded)
decoded = encoder.inverse_transform(encoded)
print("Decoded:", decoded)