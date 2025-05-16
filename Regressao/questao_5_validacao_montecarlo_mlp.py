# Regressão: 5 questão MLP

import numpy as np

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

    for _ in range(max_epochs):
        z1 = X @ W1 + b1
        a1 = sigmoid(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2
        error = y_pred - y
        dW2 = a1.T @ error / N
        db2 = np.mean(error, axis=0, keepdims=True)
        da1 = error @ W2.T * sigmoid_derivative(z1)
        dW1 = X.T @ da1 / N
        db1 = np.mean(da1, axis=0, keepdims=True)
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    return W1, b1, W2, b2

def monte_carlo_validation(X, y, hidden_neurons=10, R=250):
    mse_list = []
    for _ in range(R):
        indices = np.random.permutation(len(X))
        split = int(len(X) * 0.8)
        train_idx, test_idx = indices[:split], indices[split:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        W1, b1, W2, b2 = train_mlp(X_train, y_train, hidden_neurons)
        a1_test = sigmoid(X_test @ W1 + b1)
        y_pred_test = a1_test @ W2 + b2
        mse = np.mean((y_pred_test - y_test) ** 2)
        mse_list.append(mse)

    mse_array = np.array(mse_list)
    print(f"Média MSE   = {np.mean(mse_array):.6f}")
    print(f"Desvio-Padrão = {np.std(mse_array):.6f}")
    print(f"Maior Valor   = {np.max(mse_array):.6f}")
    print(f"Menor Valor   = {np.min(mse_array):.6f}")

monte_carlo_validation(X_norm, y_norm, hidden_neurons=10, R=250)
