# Renita Kurian - PES1UG20CS331
# Lab 7 - Adaboost

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):
        return (sum(sample_weights * (np.not_equal(y, y_pred)).astype(int)))/sum(sample_weights)

    def compute_alpha(self, error):
        eps = 1e-9
        if(error == 0):
            error = eps
        return 0.5* (np.log((1 - error) / error))

    def update_weights(self, y, y_pred, sample_weights, alpha):
        err = self.stump_error(y, y_pred, sample_weights)     
        if(err == 0):
            sample_weights= sample_weights * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))     
            return sample_weights
        for i in range(len(sample_weights)):
            if y[i] == y_pred[i]:
                sample_weights[i] = sample_weights[i]/(2*(1-err))
            else:
                sample_weights[i] = sample_weights[i]/(2*err)    
        return sample_weights


    def predict(self, X):
        p = np.array([self.stumps[model].predict(X) for model in range(self.n_stumps)])
        prediction = np.sign(p[0])
        return prediction

    def evaluate(self, X, y):
        pred = self.predict(X)
        correct = (pred == y)
        accuracy = np.mean(correct) * 100  # accuracy calculation
        return accuracy