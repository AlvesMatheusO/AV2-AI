# MULTICLASSES: 3 QUESTÃO (RBF)

import numpy as np

class RBF:
    def __init__(self, input_dim, hidden_dim, output_dim, sigma=1.0):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.centros = None
        self.w = None

    def _rbf(self, x, c):
        return np.exp(-np.linalg.norm(x - c, axis=0)**2 / (2 * self.sigma**2))

    def _calc_hidden_layer(self, X):
        G = np.zeros((self.hidden_dim, X.shape[1]))
        for i, c in enumerate(self.centros.T):
            for j in range(X.shape[1]):
                G[i, j] = self._rbf(X[:, j], c)
        return G

    def train(self, X, Y):
        indices = np.random.choice(X.shape[1], self.hidden_dim, replace=False)
        self.centros = X[:, indices]
        G = self._calc_hidden_layer(X)
        G = np.vstack((np.ones((1, G.shape[1])), G))
        self.w = np.linalg.pinv(G.T) @ Y.T

    def predict(self, X):
        G = self._calc_hidden_layer(X)
        G = np.vstack((np.ones((1, G.shape[1])), G))
        Y_out = self.w.T @ G
        return np.argmax(Y_out, axis=0)
