import numpy as np
import matplotlib.pyplot as plt

# Questão 1
data = np.loadtxt('aerogerador.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X_norm = (X - np.mean(X)) / np.std(X)
y_norm = (y - np.mean(y)) / np.std(y)
X_bias = np.hstack([np.ones_like(X_norm), X_norm])

# Questão 2
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=5, color='blue', alpha=0.6)
plt.title('Gráfico de Dispersão - Velocidade do Vento vs Potência Gerada')
plt.xlabel('Velocidade do Vento (m/s)')
plt.ylabel('Potência Gerada (kW)')
plt.grid(True)
plt.show()

# Questão 3: ADALINE
def train_adaline(X, y, learning_rate=0.01, max_epochs=1000, tol=1e-6):
    N, p = X.shape
    w = np.random.randn(p, 1) * 0.01
    mse_history = []

    for epoch in range(max_epochs):
        y_pred = X @ w
        error = y_pred - y
        mse = np.mean(error ** 2)
        mse_history.append(mse)
        grad = X.T @ error / N
        w -= learning_rate * grad
        if mse < tol:
            break

    return w, mse_history

weights, mse_history = train_adaline(X_bias, y_norm)
y_pred_norm = X_bias @ weights
y_pred = y_pred_norm * np.std(y) + np.mean(y)

# Plot do ajuste do modelo ADALINE
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=5, color='orange', label='Dados reais')
plt.plot(X, y_pred, color='red', label='Ajuste ADALINE')
plt.title('Ajuste do Modelo ADALINE - Regressão')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.legend()
plt.grid(True)
plt.show()

# Curva de aprendizado
plt.figure(figsize=(8, 4))
plt.plot(mse_history, color='green')
plt.title('Curva de Aprendizado (Erro MSE por Época)')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.grid(True)
plt.show()
