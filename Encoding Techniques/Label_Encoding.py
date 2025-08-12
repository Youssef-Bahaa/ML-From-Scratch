class LabelEncoder:
    def __init__(self):
        self.class_to_int = {}
        self.int_to_class = {}

    def fit(self,labels):
        unique = sorted(set(labels))

        for idx,val in enumerate(unique):
            self.class_to_int[val] = idx
            self.int_to_class[idx] = val

        return self

    def transform(self, labels):
        output = []
        for i,val in enumerate(labels):
            output.append(self.class_to_int[val])

        return output

    def fit_transform(self, labels):
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, ints):
        output = []
        for val in ints:
            output.append(self.int_to_class[val])

        return output




labels = ["cat", "dog", "cat", "bird", "dog"]
encoder = LabelEncoder()
lb = encoder.fit_transform(labels)
i = encoder.inverse_transform(lb)
print(lb)
print(i)