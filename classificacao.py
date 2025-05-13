import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
#  Funções auxiliares
# ---------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    return np.array([[tn, fp],
                     [fn, tp]])

# ---------------------------------------------------
#  Modelos em NumPy puro
# ---------------------------------------------------
class Perceptron:
    def __init__(self, lr=0.01, epochs=50):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y, X_val=None, y_val=None):
        N, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        self.train_hist = []
        self.val_hist = []
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = xi.dot(self.w) + self.b
                y_pred = 1 if z >= 0 else 0
                update = self.lr * (yi - y_pred)
                self.w += update * xi
                self.b += update
            # registro de acurácia
            y_tr = (X.dot(self.w) + self.b >= 0).astype(int)
            self.train_hist.append(accuracy(y, y_tr))
            if X_val is not None:
                y_va = (X_val.dot(self.w) + self.b >= 0).astype(int)
                self.val_hist.append(accuracy(y_val, y_va))

    def predict(self, X):
        return (X.dot(self.w) + self.b >= 0).astype(int)


class MLP:
    def __init__(self, nhidden=20, lr=0.02, epochs=50):
        self.nh = nhidden
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y, X_val=None, y_val=None):
        N, p = X.shape
        # inicializa pesos
        self.W1 = np.random.randn(p, self.nh) * 0.1
        self.b1 = np.zeros(self.nh)
        self.W2 = np.random.randn(self.nh) * 0.1
        self.b2 = 0.0
        self.train_hist = []
        self.val_hist = []

        for _ in range(self.epochs):
            # forward
            Z1 = np.tanh(X.dot(self.W1) + self.b1)
            A2 = sigmoid(Z1.dot(self.W2) + self.b2)
            # backprop
            error = A2 - y
            dW2 = Z1.T.dot(error * A2 * (1 - A2))
            db2 = np.sum(error * A2 * (1 - A2))
            dZ1 = (error * A2 * (1 - A2))[:, None] * self.W2[None, :] * (1 - Z1**2)
            dW1 = X.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0)
            # atualiza
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            # registra acurácia
            y_tr = (sigmoid(np.tanh(X.dot(self.W1) + self.b1).dot(self.W2) + self.b2) >= 0.5).astype(int)
            self.train_hist.append(accuracy(y, y_tr))
            if X_val is not None:
                y_va = (sigmoid(np.tanh(X_val.dot(self.W1) + self.b1).dot(self.W2) + self.b2) >= 0.5).astype(int)
                self.val_hist.append(accuracy(y_val, y_va))

    def predict(self, X):
        Z1 = np.tanh(X.dot(self.W1) + self.b1)
        return (sigmoid(Z1.dot(self.W2) + self.b2) >= 0.5).astype(int)


class RBFNetwork:
    def __init__(self, n_centers=15, lr=0.02, epochs=50):
        self.k = n_centers
        self.lr = lr
        self.epochs = epochs
        self.sigma = None

    def _rbf_design(self, X):
        diff = X[:, None, :] - self.centers[None, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
        return np.exp(-(D**2) / (2 * self.sigma**2))

    def fit(self, X, y, X_val=None, y_val=None):
        N = X.shape[0]
        # escolhe centros
        idx = np.random.choice(N, self.k, replace=False)
        self.centers = X[idx]
        # calcula sigma
        cd = self.centers[:, None, :] - self.centers[None, :, :]
        pd = np.sqrt(np.sum(cd**2, axis=2))
        self.sigma = np.mean(pd[pd > 0])
        # design matrix
        Phi = self._rbf_design(X)
        Phi = np.hstack([np.ones((N, 1)), Phi])
        # inicializa pesos
        self.W = np.random.randn(self.k + 1) * 0.1

        self.train_hist = []
        self.val_hist = []

        for _ in range(self.epochs):
            A = sigmoid(Phi.dot(self.W))
            error = y - A
            grad = Phi.T.dot(error * A * (1 - A))
            self.W += self.lr * grad
            # registra acurácia
            preds_tr = (A >= 0.5).astype(int)
            self.train_hist.append(accuracy(y, preds_tr))
            if X_val is not None:
                Phi_val = np.hstack([np.ones((X_val.shape[0], 1)),
                                     self._rbf_design(X_val)])
                A_val = sigmoid(Phi_val.dot(self.W))
                preds_va = (A_val >= 0.5).astype(int)
                self.val_hist.append(accuracy(y_val, preds_va))

    def predict(self, X):
        Phi = np.hstack([np.ones((X.shape[0], 1)), self._rbf_design(X)])
        return (sigmoid(Phi.dot(self.W)) >= 0.5).astype(int)


# ---------------------------------------------------
#  Função principal
# ---------------------------------------------------
def main():
    print(">>> Iniciando classificação Spiral3d")

    # 1. Carrega e organiza
    data = np.loadtxt("Spiral3d.csv", delimiter=",")
    X = data[:, :3]
    y = data[:, 3].astype(int)
    N, p = X.shape
    print(f"Dimensões: X={X.shape}, y={y.shape}")

    # 2. Visualização inicial (pairwise 2D)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (i, j) in zip(axes, [(0,1), (0,2), (1,2)]):
        sns.scatterplot(x=X[:, i], y=X[:, j], hue=y,
                        palette="bwr", edgecolor="k", alpha=0.7, ax=ax)
        ax.set_xlabel(f"X{i+1}"); ax.set_ylabel(f"X{j+1}")
    plt.suptitle("Projeções 2D das 3 variáveis", y=1.02)
    plt.tight_layout(); plt.show()

    # 3. Instancia modelos base
    percep   = Perceptron(lr=0.01, epochs=50)
    mlp_base = MLP(nhidden=20, lr=0.02, epochs=50)
    rbf_base = RBFNetwork(n_centers=15, lr=0.02, epochs=50)

    # 4. Underfit vs Overfit (split 80/20)
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    Xtr, ytr = X[perm[:split]], y[perm[:split]]
    Xte, yte = X[perm[split:]], y[perm[split:]]

    # MLP small vs large
    mlp_u = MLP(nhidden=2,  lr=0.05, epochs=50); mlp_u.fit(Xtr, ytr, Xte, yte)
    mlp_o = MLP(nhidden=50, lr=0.01, epochs=50); mlp_o.fit(Xtr, ytr, Xte, yte)
    print(f"MLP2  → Train={mlp_u.train_hist[-1]:.3f}, Val={mlp_u.val_hist[-1]:.3f}")
    print(f"MLP50 → Train={mlp_o.train_hist[-1]:.3f}, Val={mlp_o.val_hist[-1]:.3f}")

    # RBF few vs many centers
    rbf_u = RBFNetwork(n_centers=5,  lr=0.05, epochs=50); rbf_u.fit(Xtr, ytr, Xte, yte)
    rbf_o = RBFNetwork(n_centers=30, lr=0.01, epochs=50); rbf_o.fit(Xtr, ytr, Xte, yte)
    print(f"RBF5  → Train={rbf_u.train_hist[-1]:.3f}, Val={rbf_u.val_hist[-1]:.3f}")
    print(f"RBF30 → Train={rbf_o.train_hist[-1]:.3f}, Val={rbf_o.val_hist[-1]:.3f}")

    # Curvas de aprendizado
    plt.figure(figsize=(12,4))
    # MLP
    plt.subplot(1,2,1)
    plt.plot(mlp_u.train_hist, "--", label="MLP2 Train")
    plt.plot(mlp_u.val_hist,   "-",  label="MLP2 Val")
    plt.plot(mlp_o.train_hist, "--", label="MLP50 Train")
    plt.plot(mlp_o.val_hist,   "-",  label="MLP50 Val")
    plt.title("MLP Learning Curves"); plt.legend()
    # RBF
    plt.subplot(1,2,2)
    plt.plot(rbf_u.train_hist, "--", label="RBF5 Train")
    plt.plot(rbf_u.val_hist,   "-",  label="RBF5 Val")
    plt.plot(rbf_o.train_hist, "--", label="RBF30 Train")
    plt.plot(rbf_o.val_hist,   "-",  label="RBF30 Val")
    plt.title("RBF Learning Curves"); plt.legend()
    plt.tight_layout(); plt.show()

    # 5. Monte Carlo (R = 250)
    R = 250
    models = {"Perceptron": percep, "MLP": mlp_base, "RBF": rbf_base}
    results = {m: {"acc":[], "sens":[], "spec":[]} for m in models}

    for r in range(R):
        perm = np.random.permutation(N)
        Xr, yr = X[perm], y[perm]
        s = int(0.8 * N)
        Xtr, ytr = Xr[:s], yr[:s]
        Xte, yte = Xr[s:], yr[s:]
        for name, mod in models.items():
            mod.fit(Xtr, ytr)  # aqui não usamos validação
            yp = mod.predict(Xte)
            tn, fp, fn, tp = confusion_matrix_manual(yte, yp).ravel()
            results[name]["acc" ].append((tp+tn)/(tp+tn+fp+fn))
            results[name]["sens"].append(tp/(tp+fn) if tp+fn>0 else 0)
            results[name]["spec"].append(tn/(tn+fp) if tn+fp>0 else 0)

    # 6. Melhor e Pior rodada
    mean_acc = np.mean([results[m]["acc"] for m in models], axis=0)
    best_i, worst_i = np.argmax(mean_acc), np.argmin(mean_acc)

    for label, idx_case in [("Melhor", best_i), ("Pior", worst_i)]:
        print(f"\n=== {label} rodada (r={idx_case}) ===")
        fig, axes = plt.subplots(1, len(models), figsize=(12,4))
        for ax, (name, mod) in zip(axes, models.items()):
            perm = np.random.RandomState(idx_case).permutation(N)
            Xr, yr = X[perm], y[perm]
            s = int(0.8 * N)
            Xtr, ytr = Xr[:s], yr[:s]
            Xte, yte = Xr[s:], yr[s:]
            mod.fit(Xtr, ytr)
            cm = confusion_matrix_manual(yte, mod.predict(Xte))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False)
            ax.set_title(name)
        plt.tight_layout(); plt.show()

        # curvas de aprendizado
        plt.figure(figsize=(6,4))
        for name, mod in models.items():
            if hasattr(mod, "train_hist"):
                plt.plot(mod.train_hist, "--", label=f"{name} train")
                plt.plot(mod.val_hist,            "-",  label=f"{name} val")
        plt.title(f"Curvas de aprendizado ({label})")
        plt.xlabel("Época"); plt.ylabel("Acurácia")
        plt.legend(); plt.tight_layout(); plt.show()

    # 7. Estatísticas finais
    for met, title in [("acc","Acurácia"),("sens","Sensibilidade"),("spec","Especificidade")]:
        vals = np.array([results[name][met] for name in models])
        means = vals.mean(axis=1); stds = vals.std(axis=1)
        maxs = vals.max(axis=1); mins = vals.min(axis=1)

        fig, ax = plt.subplots(); ax.axis("off")
        cellText = np.vstack([means, stds, maxs, mins]).T
        tbl = ax.table(cellText=cellText,
                       rowLabels=list(models.keys()),
                       colLabels=["Média","Desv.Padrão","Máximo","Mínimo"],
                       loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.5)
        ax.set_title(f"Estatísticas de {title}"); plt.show()

        plt.figure(figsize=(6,4))
        sns.boxplot(data=[results[n][met] for n in models], palette="pastel")
        plt.xticks(ticks=range(len(models)), labels=list(models.keys()))
        plt.title(f"Distribuição de {title}"); plt.ylabel(title)
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
