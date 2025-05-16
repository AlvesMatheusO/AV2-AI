# MULTICLASSES: 3 QUESTÃO (ADALINE)

import numpy as np

class Adaline:
    def __init__(self, input_dim, learning_rate=0.01, max_epochs=1000):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.randn(input_dim) * 0.01

    def train(self, X, y):
        for _ in range(self.max_epochs):
            y_pred = self.w @ X
            error = y - y_pred
            self.w += self.learning_rate * (error @ X.T) / X.shape[1]

    def predict(self, X):
        return np.sign(self.w @ X)

def train_adaline_multiclass(X, Y, learning_rate=0.01, epochs=1000):
    classifiers = []
    for i in range(Y.shape[0]):
        model = Adaline(input_dim=X.shape[0], learning_rate=learning_rate, max_epochs=epochs)
        model.train(X, Y[i])
        classifiers.append(model)
    return classifiers

def predict_adaline_multiclass(models, X):
    outputs = [model.w @ X for model in models]
    outputs = np.array(outputs)
    preds = np.argmax(outputs, axis=0)
    return preds
