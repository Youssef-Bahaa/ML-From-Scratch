import numpy as np


class KMeans:
    def __init__(self,k=3,max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels_ = None

    def fit(self,X):
        n_samples , n_features = X.shape

        random_index = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_index]

        for _ in range(self.max_iter):
            clusters = self.assign_clusters(X)

            new_centroids = self.update_centroids(X,clusters)

            if np.allclose(self.centroids ,new_centroids,self.tol ):
                break

            self.centroids = new_centroids

        self.labels_ = self.assign_clusters(X)

    def assign_clusters(self, X):
        clusters = []
        for point in X:
            distances = [np.sum((point - centroid) ** 2) for centroid in self.centroids]
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def update_centroids(self,X,clusters):
        new_centroids =[]
        for i in range(self.k):
            Cluster_points = X[clusters == i]
            if len(Cluster_points) > 0:
                avg = np.mean(Cluster_points,axis = 0)
            else :
                avg = self.centroids[i]
            new_centroids.append(avg)

        return np.array(new_centroids)


    def predict(self,X):
        return self.assign_clusters(X)

