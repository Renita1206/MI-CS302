#Renita Kurian - PES1UG20CS331
# Week 4 - KNN Lab Assignment

import numpy as np
from math import *
from decimal import Decimal

class KNN:

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        self.data = data
        self.target = target.astype(np.int64)
        return self
        
    def my_p_root(self, value, root):
        my_root_value = 1 / float(root)
        return round (Decimal(value) ** Decimal(my_root_value), 3)
        
    def my_minkowski_distance(self, x, y, p_value):
        return float(self.my_p_root(sum(pow(abs(m-n), p_value) for m, n in zip(x, y)), p_value))	
        
    def find_distance(self, x):
        a = []
        for i in range(x.shape[0]):
            d = x[i]
            d1 = []
            for j in range(self.data.shape[0]): 
                n = self.data[j]
                d1.append(self.my_minkowski_distance(d, n, self.p))
            a.append(d1)
        return a

    def k_neighbours(self, x):
        lni = self.find_distance(x)
        r = [[], []]
        
        for i in range(len(lni)):
            indices = [i for i in range(self.data.shape[0])]
            d = list(list(zip(*list(sorted(zip(lni[i], indices)))))[0])
            e = list(list(zip(*list(sorted(zip(lni[i], indices)))))[1])
            r[0].append(d[0:self.k_neigh])
            r[1].append(e[0:self.k_neigh])
        return r
            
    def predict(self, x):
        indices = self.k_neighbours(x)[1]
        r = []
        for i in range(len(indices)):
            f = {}
            for j in range(len(indices[i])):
                if self.target[indices[i][j]] in f:
                    f[self.target[indices[i][j]]] += 1
                else:
                    f[self.target[indices[i][j]]] = 1 
            maxF = 0
            maxK = None
            for i in range(min(f), max(f)+1):
                if f[i] > maxF:
                    maxF = f[i]
                    maxK = i
            r.append(maxK)
        return r
	

    def evaluate(self, x, y):
        pred=self.predict(x)
        right=np.sum(pred==y)
        return(100*((right)/len(y)))
