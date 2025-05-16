# MULTICLASSES: 3 QUESTÃO (MLP)

import numpy as np

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, max_epochs=1000):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.w2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_deriv(self, x):
        return 1.0 - np.tanh(x) ** 2

    def train(self, X, Y):
        for _ in range(self.max_epochs):
            z1 = self.w1 @ X + self.b1
            a1 = self.tanh(z1)
            z2 = self.w2 @ a1 + self.b2
            a2 = z2

            dz2 = Y - a2
            dw2 = dz2 @ a1.T / X.shape[1]
            db2 = np.mean(dz2, axis=1, keepdims=True)

            dz1 = (self.w2.T @ dz2) * self.tanh_deriv(z1)
            dw1 = dz1 @ X.T / X.shape[1]
            db1 = np.mean(dz1, axis=1, keepdims=True)

            self.w1 += self.lr * dw1
            self.b1 += self.lr * db1
            self.w2 += self.lr * dw2
            self.b2 += self.lr * db2

    def predict(self, X):
        a1 = self.tanh(self.w1 @ X + self.b1)
        a2 = self.w2 @ a1 + self.b2
        return np.argmax(a2, axis=0)

# mlp = MLP(input_dim=7, hidden_dim=10, output_dim=3, learning_rate=0.01, max_epochs=1000)
# mlp.train(X_train, Y_train)  # onde Y_train é Y com valores reais, não one-hot bipolar
# preds = mlp.predict(X_test)