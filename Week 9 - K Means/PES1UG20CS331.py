# Renita Kurian - PES1UG20CS331
# Week 9 - K Means

import numpy as np

class KMeansClustering:

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids
        return self

    def e_step(self, data):
        d = np.zeros((self.n_cluster, data.shape[0]))
        i = 0
        for center in self.centroids:
            j = 0
            for point in data:
                dist = np.linalg.norm(center - point)
                d[i][j] = dist
                j += 1
            i += 1

        result = np.zeros(data.shape[0])
        for index_i in range(data.shape[0]):
            min_center = 0
            for index_j in range(self.n_cluster):
                if d[index_j][index_i] < d[min_center][index_i]:
                    min_center = index_j
            result[index_i] = min_center
        return result

    def m_step(self, data, cluster_assgn):
        centroids = np.zeros((self.n_cluster, data.shape[1]))
        sum = np.zeros(self.n_cluster)
        for index in range(len(cluster_assgn)):
            centroids[cluster_assgn[index]] += data[index]
            sum[cluster_assgn[index]] += 1
        for i in range(self.n_cluster):
            centroids[i] /= sum[i]
        self.centroids = centroids

    def evaluate(self, data):
        metric = np.sum((data[:, None]-self.centroids)**2)
        return metric