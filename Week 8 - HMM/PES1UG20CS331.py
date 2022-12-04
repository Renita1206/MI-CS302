# Renita Kurian - PES1UG20CS331
# Lab 8 - HMM

import numpy as np

class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(zip(self.emissions, list(range(self.M))))

    def init_x (self, x, seq):
        for i in range(self.N):
            x[0, i] = self.pi[i] * self.B[i, self.emissions_dict[seq[0]]]
        return x

    def recx(self, seq, x):
        for i in range(1, len(seq)):
            for j in range(self.N):
                atTmax = -1
                for k in range(self.N): 
                    if x[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]] > atTmax:
                        atTmax = x[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]]
                x[i, j] = atTmax
        return x

    def maybe_recx(self, seq, x, maybe_x):
        for i in range(1, len(seq)):
            for j in range(self.N):
                atTmax = -1
                max_x = -1
                for k in range(self.N):
                    if x[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]] > atTmax:
                        atTmax = x[i - 1, k] * self.A[k, j] * self.B[j, self.emissions_dict[seq[i]]]
                        max_x = k
                maybe_x[i, j] = max_x
        return maybe_x

    def termination(self, seq, x, max_x):
        atTmax = -1
        for i in range(self.N):
            if x[len(seq) - 1, i] > atTmax:
                atTmax = x[len(seq) - 1, i]
                max_x = i
        return max_x

    def viterbi_algorithm(self, seq):
        x = np.zeros((len(seq), self.N))
        x = self.init_x(x, seq)

        maybe_x = np.zeros((len(seq), self.N), dtype=int)        
        for i in range(self.N):
            maybe_x[0, i] = 0
        
        x = self.recx(seq, x)
        maybe_x = self.maybe_recx(seq, x, maybe_x)
        
        max_x = -1
        max_x = self.termination(seq, x, max_x)
        res = [max_x]
        
        for i in range(len(seq) - 1, 0, -1):
            res.append(maybe_x[i, res[-1]])
        res.reverse()

        hidden_states_sequence = {x : y for y, x in self.states_dict.items()}
        return [hidden_states_sequence[i] for i in res]