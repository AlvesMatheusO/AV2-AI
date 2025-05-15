import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Funções de métrica
# ----------------------
def compute_metrics(y_true, y_pred, C):
    """
    Retorna (acc, sens, spec, CM) para classificação multi-classe com macro-averaging.
    """
    CM = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        CM[t, p] += 1
    acc = np.trace(CM) / CM.sum()
    sens_list, spec_list = [], []
    for c in range(C):
        TP = CM[c,c]
        FN = CM[c,:].sum() - TP
        FP = CM[:,c].sum() - TP
        TN = CM.sum() - TP - FN - FP
        sens = TP/(TP+FN) if TP+FN>0 else 0
        spec = TN/(TN+FP) if TN+FP>0 else 0
        sens_list.append(sens)
        spec_list.append(spec)
    return acc, np.mean(sens_list), np.mean(spec_list), CM

# ----------------------
# Modelos
# ----------------------
class Adaline:
    def __init__(self, lr=0.01, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs

    def fit(self, X, Y, X_val=None, Y_val=None):
        # X: N×p, Y: N×C with values +1 or -1
        N, p = X.shape
        C = Y.shape[1]
        self.W = np.random.randn(p, C) * 0.1
        self.b = np.zeros(C)
        # históricos
        self.train_hist = {'acc':[], 'sens':[], 'spec':[]}
        self.val_hist   = {'acc':[], 'sens':[], 'spec':[]}
        for ep in range(self.max_epochs):
            # predição linear
            Y_pred = X.dot(self.W) + self.b  # N×C
            err = Y - Y_pred
            # ajuste pelos mínimos quadrados (LMS)
            self.W += self.lr * (X.T.dot(err) / N)
            self.b += self.lr * err.mean(axis=0)
            # métricas de treino
            y_tr_pred = np.argmax(Y_pred, axis=1)
            y_tr_true = np.argmax(Y, axis=1)
            acc, sens, spec, _ = compute_metrics(y_tr_true, y_tr_pred, C)
            self.train_hist['acc'].append(acc)
            self.train_hist['sens'].append(sens)
            self.train_hist['spec'].append(spec)
            # validação se dados fornecidos
            if X_val is not None:
                Yv_pred = X_val.dot(self.W) + self.b
                y_v_pred = np.argmax(Yv_pred, axis=1)
                y_v_true = np.argmax(Y_val, axis=1)
                acc, sens, spec, _ = compute_metrics(y_v_true, y_v_pred, C)
                self.val_hist['acc'].append(acc)
                self.val_hist['sens'].append(sens)
                self.val_hist['spec'].append(spec)

    def predict(self, X):
        scores = X.dot(self.W) + self.b
        return np.argmax(scores, axis=1)

class MLP:
    def __init__(self, layers=[6,20,3], activation='tanh', lr=0.1, max_epochs=200):
        """
        layers: [p, h1, ..., hL, C]
        activation: 'tanh' ou 'sigmoid'
        """
        self.layers = layers
        self.lr = lr
        self.max_epochs = max_epochs
        if activation=='tanh':
            self.act = np.tanh
            self.act_deriv = lambda a: 1 - a**2
        else:
            self.act = lambda z: 1/(1+np.exp(-z))
            self.act_deriv = lambda a: a*(1-a)

    def fit(self, X, Y, X_val=None, Y_val=None):
        np.random.seed(0)
        N, p = X.shape
        C = Y.shape[1]
        # inicializa pesos/bias
        L = len(self.layers)-1
        self.W = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.1
                  for i in range(L)]
        self.b = [np.zeros(self.layers[i+1]) for i in range(L)]
        # históricos
        self.train_hist = {'acc':[], 'sens':[], 'spec':[]}
        self.val_hist   = {'acc':[], 'sens':[], 'spec':[]}

        for ep in range(self.max_epochs):
            # FORWARD
            A = [X]
            for i in range(L):
                Z = A[-1].dot(self.W[i]) + self.b[i]
                A.append(self.act(Z))
            # BACKPROP
            delta = (A[-1] - Y) * self.act_deriv(A[-1])
            grads_W = []
            grads_b = []
            for i in reversed(range(L)):
                grads_W.insert(0, A[i].T.dot(delta))
                grads_b.insert(0, delta.sum(axis=0))
                if i>0:
                    delta = delta.dot(self.W[i].T) * self.act_deriv(A[i])
            # UPDATE
            for i in range(L):
                self.W[i] -= self.lr * grads_W[i] / N
                self.b[i] -= self.lr * grads_b[i] / N
            # MÉTRICAS
            y_tr_pred = self.predict(X)
            y_tr_true = np.argmax(Y, axis=1)
            acc, sens, spec, _ = compute_metrics(y_tr_true, y_tr_pred, C)
            self.train_hist['acc'].append(acc)
            self.train_hist['sens'].append(sens)
            self.train_hist['spec'].append(spec)
            if X_val is not None:
                y_v_pred = self.predict(X_val)
                y_v_true = np.argmax(Y_val, axis=1)
                acc, sens, spec, _ = compute_metrics(y_v_true, y_v_pred, C)
                self.val_hist['acc'].append(acc)
                self.val_hist['sens'].append(sens)
                self.val_hist['spec'].append(spec)

    def predict(self, X):
        A = X
        for W,b in zip(self.W, self.b):
            A = self.act(A.dot(W)+b)
        return np.argmax(A, axis=1)

class RBFNetwork:
    def __init__(self, n_centers=15, lr=0.01, max_epochs=100):
        self.k = n_centers
        self.lr = lr
        self.max_epochs = max_epochs

    def _rbf_design(self, X):
        diff = X[:,None,:] - self.centers[None,:,:]
        D = np.sqrt((diff**2).sum(axis=2))
        return np.exp(- (D**2) / (2*self.sigma**2))

    def fit(self, X, Y, X_val=None, Y_val=None):
        N, p = X.shape
        C = Y.shape[1]
        # centros
        idx = np.random.choice(N, self.k, replace=False)
        self.centers = X[idx]
        # sigma
        pd = np.sqrt(((self.centers[:,None,:] - self.centers[None,:,:])**2).sum(2))
        self.sigma = pd[pd>0].mean()
        # design matrix
        Phi = self._rbf_design(X)
        Phi = np.hstack([np.ones((N,1)), Phi])
        # pesos
        self.W = np.random.randn(self.k+1, C) * 0.1
        # históricos
        self.train_hist = {'acc':[], 'sens':[], 'spec':[]}
        self.val_hist   = {'acc':[], 'sens':[], 'spec':[]}

        for ep in range(self.max_epochs):
            Y_pred = Phi.dot(self.W)            # N×C
            error  = Y - Y_pred
            # LMS update
            self.W += self.lr * (Phi.T.dot(error) / N)
            # métricas treino
            y_tr = np.argmax(Y_pred, axis=1)
            y_tr_true = np.argmax(Y, axis=1)
            acc, sens, spec, _ = compute_metrics(y_tr_true, y_tr, C)
            self.train_hist['acc'].append(acc)
            self.train_hist['sens'].append(sens)
            self.train_hist['spec'].append(spec)
            # validação
            if X_val is not None:
                Phi_v = np.hstack([np.ones((X_val.shape[0],1)),
                                   self._rbf_design(X_val)])
                Yv_pred = Phi_v.dot(self.W)
                y_v = np.argmax(Yv_pred, axis=1)
                y_v_true = np.argmax(Y_val, axis=1)
                acc, sens, spec, _ = compute_metrics(y_v_true, y_v, C)
                self.val_hist['acc'].append(acc)
                self.val_hist['sens'].append(sens)
                self.val_hist['spec'].append(spec)

    def predict(self, X):
        Phi = np.hstack([np.ones((X.shape[0],1)),
                         self._rbf_design(X)])
        return np.argmax(Phi.dot(self.W), axis=1)

# ----------------------
# Main
# ----------------------
def main():
    # 1. Carrega dados
    data = np.loadtxt("coluna_vertebral.csv", delimiter=",", dtype=str)
    X = data[:, :6].astype(float)
    labels = data[:, 6]  # strings: "NO","DH","SL"
    N = X.shape[0]

    # 2. One-hot +1/–1
    mapping = {"NO":0, "DH":1, "SL":2}
    y_idx = np.array([mapping[l] for l in labels])
    C = 3
    Y = -np.ones((N, C), dtype=int)
    for i, c in enumerate(y_idx):
        Y[i, c] = +1

    # normalize X
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    X = (X - mu) / sigma

    # Monte Carlo
    R = 100
    results = { "Adaline":[] , "MLP":[] , "RBF":[] }
    best_case = { "idx":None, "acc":-np.inf }
    worst_case= { "idx":None, "acc": np.inf }

    for r in range(R):
        perm = np.random.permutation(N)
        s = int(0.8 * N)
        Xi, Yi = X[perm[:s]], Y[perm[:s]]
        Xv, Yv = X[perm[s:]], Y[perm[s:]]

        # instanciar modelos
        ada = Adaline(lr=0.01, max_epochs=100)
        mlp = MLP(layers=[6,20,3], activation='tanh', lr=0.1, max_epochs=200)
        rbf = RBFNetwork(n_centers=15, lr=0.01, max_epochs=100)

        for name, mod in [("Adaline",ada),("MLP",mlp),("RBF",rbf)]:
            # treina só com treino (não precisamos de curva de val aqui)
            mod.fit(Xi, Yi)
            y_pred = mod.predict(Xv)
            y_true = np.argmax(Yv, axis=1)
            acc, sens, spec, _ = compute_metrics(y_true, y_pred, C)
            results[name].append((acc, sens, spec))

        # média das 3 acurácias
        avg_acc = np.mean([results[m][-1][0] for m in results])
        if avg_acc > best_case["acc"]:
            best_case.update(idx=r, acc=avg_acc)
        if avg_acc < worst_case["acc"]:
            worst_case.update(idx=r, acc=avg_acc)

    # 5. Matriz de confusão e curvas (best e worst)
    for label, case in [("Melhor", best_case), ("Pior", worst_case)]:
        r = case["idx"]
        perm = np.random.RandomState(r).permutation(N)
        s = int(0.8 * N)
        Xi, Yi = X[perm[:s]], Y[perm[:s]]
        Xv, Yv = X[perm[s:]], Y[perm[s:]]
        fig, axes = plt.subplots(1,3, figsize=(15,4), 
                                 subplot_kw={"xticks":[0,1,2], "yticks":[0,1,2]})
        fig.suptitle(f"{label} rodada (r={r})", y=1.05)
        for ax, (name, mod) in zip(axes, [("Adaline",ada),("MLP",mlp),("RBF",rbf)]):
            # refit
            mod.fit(Xi, Yi, Xv, Yv)
            y_pred = mod.predict(Xv)
            y_true = np.argmax(Yv, axis=1)
            _, _, _, CM = compute_metrics(y_true, y_pred, C)
            sns.heatmap(CM, annot=True, fmt="d", ax=ax, cbar=False,
                        cmap="Blues", square=True)
            ax.set_title(name)
            ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
        plt.tight_layout(); plt.show()

        # curvas de aprendizado
        plt.figure(figsize=(6,4))
        for name, mod in [("Adaline",ada),("MLP",mlp),("RBF",rbf)]:
            plt.plot(mod.train_hist['acc'], '--', label=f"{name} Train")
            plt.plot(mod.val_hist['acc'],   '-',  label=f"{name} Val")
        plt.title(f"Curvas de Aprendizado ({label})")
        plt.xlabel("Época"); plt.ylabel("Acurácia")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 6. Estatísticas finais (acurácia)
    import pandas as pd  # apenas para a tabela
    stats = {}
    for name in results:
        arr = np.array(results[name])[:,0]  # só acc
        stats[name] = {
            "Média": arr.mean(),
            "Desvio-Padrão": arr.std(),
            "Máximo": arr.max(),
            "Mínimo": arr.min()
        }
    df = pd.DataFrame(stats).T
    print("\n=== Estatísticas de Acurácia (100 rodadas) ===")
    display(df)

    # boxplot
    plt.figure(figsize=(6,4))
    sns.boxplot(data=[np.array(results[n])[:,0] for n in results], palette="Set2")
    plt.xticks(range(3), list(results.keys()))
    plt.title("Distribuição de Acurácias")
    plt.ylabel("Acurácia")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
