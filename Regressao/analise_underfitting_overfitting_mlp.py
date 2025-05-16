# Regressão: 4 questão MLP

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('aerogerador.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X_norm = (X - np.mean(X)) / np.std(X)
y_norm = (y - np.mean(y)) / np.std(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def train_mlp(X, y, hidden_neurons=10, learning_rate=0.01, max_epochs=300):
    N, input_dim = X.shape
    output_dim = y.shape[1]
    W1 = np.random.randn(input_dim, hidden_neurons) * 0.1
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))
    mse_history = []
    for epoch in range(max_epochs):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        dW2 = a1.T @ error / N
        db2 = np.mean(error, axis=0, keepdims=True)
        da1 = error @ W2.T * sigmoid_derivative(z1)
        dW1 = X.T @ da1 / N
        db1 = np.mean(da1, axis=0, keepdims=True)
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
    return mse_history

hidden_neurons_sub = 2
hidden_neurons_normal = 10
hidden_neurons_super = 50

mse_sub = train_mlp(X_norm, y_norm, hidden_neurons=hidden_neurons_sub)
mse_normal = train_mlp(X_norm, y_norm, hidden_neurons=hidden_neurons_normal)
mse_super = train_mlp(X_norm, y_norm, hidden_neurons=hidden_neurons_super)

plt.figure(figsize=(10, 5))
plt.plot(mse_sub, label=f'Subdimensionado ({hidden_neurons_sub} neurônios)', color='red')
plt.plot(mse_normal, label=f'Normal ({hidden_neurons_normal} neurônios)', color='green')
plt.plot(mse_super, label=f'Superdimensionado ({hidden_neurons_super} neurônios)', color='blue')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.title('Curva de Aprendizado - MLP (Underfitting x Overfitting)')
plt.legend()
plt.grid(True)
plt.show()
