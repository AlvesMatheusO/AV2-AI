import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def sensitivity(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def specificity(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def confusion_matrix_manual(y_true, y_pred):
    tp = np.sum((y_true==1)&(y_pred==1))
    tn = np.sum((y_true==0)&(y_pred==0))
    fp = np.sum((y_true==0)&(y_pred==1))
    fn = np.sum((y_true==1)&(y_pred==0))
    return np.array([[tn, fp],
                     [fn, tp]])



#  Perceptron Simples 

class Perceptron:
    def __init__(self, lr=0.01, max_epochs=100):
        self.lr = lr
        self.max_epochs = max_epochs

    def fit(self, X, y, X_val=None, y_val=None):
        N, p = X.shape
        self.w = np.zeros(p)
        self.b = 0.0
        self.train_hist = {'acc': [], 'sens': [], 'spec': []}
        self.val_hist   = {'acc': [], 'sens': [], 'spec': []}

        for ep in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - (xi.dot(self.w)+self.b >= 0))
                if update != 0:
                    errors += 1
                    self.w += update * xi
                    self.b += update
            # métricas de treino
            y_tr = self.predict(X)
            self.train_hist['acc'].append(accuracy(y, y_tr))
            self.train_hist['sens'].append(sensitivity(y, y_tr))
            self.train_hist['spec'].append(specificity(y, y_tr))
            # métricas de validação
            if X_val is not None:
                y_va = self.predict(X_val)
                self.val_hist['acc'].append(accuracy(y_val, y_va))
                self.val_hist['sens'].append(sensitivity(y_val, y_va))
                self.val_hist['spec'].append(specificity(y_val, y_va))
            if errors == 0:
                break

    def predict(self, X):
        return (X.dot(self.w) + self.b >= 0).astype(int)



#  MLP Genérico 

class MLP:
    def __init__(self, layers=[3,50,20,1], activation='tanh', lr=0.1, max_epochs=200):
        """
        layers: [input_dim, hidden1, hidden2, ..., output_dim]
        activation: 'tanh' ou 'sigmoid'
        """
        self.layers = layers
        self.activation = activation
        self.lr = lr
        self.max_epochs = max_epochs
        if activation == 'tanh':
            self.act, self.act_deriv = tanh, tanh_derivative
        else:
            self.act, self.act_deriv = sigmoid, sigmoid_derivative

    def fit(self, X, y, X_val=None, y_val=None):
        np.random.seed(0)
        # inicializa pesos e biases
        self.W = []
        self.b = []
        for i in range(len(self.layers)-1):
            self.W.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.1)
            self.b.append(np.zeros(self.layers[i+1]))
        self.train_hist = {'acc': [], 'sens': [], 'spec': []}
        self.val_hist   = {'acc': [], 'sens': [], 'spec': []}

        for ep in range(self.max_epochs):
            # forward
            A = [X]
            for W, b in zip(self.W, self.b):
                Z = A[-1].dot(W) + b
                A.append(self.act(Z))
            # backprop
            delta = (A[-1] - y.reshape(-1,1)) * self.act_deriv(A[-1])
            dW = [A[-2].T.dot(delta)]
            db = [delta.sum(axis=0)]
            for l in range(len(self.layers)-3, -1, -1):
                delta = delta.dot(self.W[l+1].T) * self.act_deriv(A[l+1])
                dW.insert(0, A[l].T.dot(delta))
                db.insert(0, delta.sum(axis=0))
            # atualização
            for i in range(len(self.W)):
                self.W[i] -= self.lr * dW[i]
                self.b[i] -= self.lr * db[i]
            # métricas de treino
            y_tr = self.predict(X)
            self.train_hist['acc'].append(accuracy(y, y_tr))
            self.train_hist['sens'].append(sensitivity(y, y_tr))
            self.train_hist['spec'].append(specificity(y, y_tr))
            # métricas de validação
            if X_val is not None:
                y_va = self.predict(X_val)
                self.val_hist['acc'].append(accuracy(y_val, y_va))
                self.val_hist['sens'].append(sensitivity(y_val, y_va))
                self.val_hist['spec'].append(specificity(y_val, y_va))

    def predict(self, X):
        A = X
        for W, b in zip(self.W, self.b):
            A = self.act(A.dot(W) + b)
        if self.activation == 'tanh':
            return (A >= 0).astype(int)
        else:
            return (A >= 0.5).astype(int)



#  RBF Network

class RBFNetwork:
    def __init__(self, n_centers=15, lr=0.02, max_epochs=100):
        self.k = n_centers
        self.lr = lr
        self.max_epochs = max_epochs
        self.sigma = None

    def _rbf_design(self, X):
        diff = X[:,None,:] - self.centers[None,:,:]
        D = np.sqrt((diff**2).sum(axis=2))
        return np.exp(- (D**2) / (2*self.sigma**2))

    def fit(self, X, y, X_val=None, y_val=None):
        N, p = X.shape
        idx = np.random.choice(N, self.k, replace=False)
        self.centers = X[idx]
        pd = np.sqrt(((self.centers[:,None,:] - self.centers[None,:,:])**2).sum(2))
        self.sigma = pd[pd>0].mean()
        Phi = self._rbf_design(X)
        Phi = np.hstack([np.ones((N,1)), Phi])
        self.W = np.random.randn(self.k+1) * 0.1
        self.train_hist = {'acc': [], 'sens': [], 'spec': []}
        self.val_hist   = {'acc': [], 'sens': [], 'spec': []}

        for ep in range(self.max_epochs):
            A = sigmoid(Phi.dot(self.W))
            error = y - A
            grad = Phi.T.dot(error * A * (1-A))
            self.W += self.lr * grad
            preds_tr = (A>=0.5).astype(int)
            self.train_hist['acc'].append(accuracy(y,preds_tr))
            self.train_hist['sens'].append(sensitivity(y,preds_tr))
            self.train_hist['spec'].append(specificity(y,preds_tr))
            if X_val is not None:
                Phi_val = np.hstack([np.ones((X_val.shape[0],1)),
                                     self._rbf_design(X_val)])
                A_val = sigmoid(Phi_val.dot(self.W))
                preds_va = (A_val>=0.5).astype(int)
                self.val_hist['acc'].append(accuracy(y_val,preds_va))
                self.val_hist['sens'].append(sensitivity(y_val,preds_va))
                self.val_hist['spec'].append(specificity(y_val,preds_va))

    def predict(self, X):
        Phi = np.hstack([np.ones((X.shape[0],1)), self._rbf_design(X)])
        return (sigmoid(Phi.dot(self.W))>=0.5).astype(int)


def main():
    print(">>> Iniciando classificação Spiral3d")

    # 1. Leitura, recodificação e normalização
    data = np.loadtxt("Spiral3d.csv", delimiter=",")
    X_raw = data[:, :3]
    y_raw = data[:, 3]
    u = np.unique(y_raw)
    if set(u) == {1,2}:
        y = (y_raw - 1).astype(int)
    elif set(u) == {-1,1}:
        y = (y_raw == 1).astype(int)
    else:
        y = y_raw.astype(int)
    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0)
    X = (X_raw - mu) / sigma
    N, p = X.shape
    print(f"Dimensões: X={X.shape}, y={y.shape}")
    print(f"Distribuição: {np.bincount(y)}")

    # 2. Projeções 2D
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for ax,(i,j) in zip(axes, [(0,1),(0,2),(1,2)]):
        sns.scatterplot(x=X[:,i], y=X[:,j], hue=y,
                        palette="bwr", edgecolor="k", ax=ax)
        ax.set_xlabel(f"X{i+1}"); ax.set_ylabel(f"X{j+1}")
    plt.suptitle("Projeções 2D das 3 variáveis", y=1.02)
    plt.tight_layout()
    plt.show()

    # 3. Instanciação de modelos
    percep   = Perceptron(lr=0.01, max_epochs=100)
    mlp_tanh = MLP(layers=[p,50,20,1], activation='tanh', lr=0.1, max_epochs=200)
    rbf_net  = RBFNetwork(n_centers=15, lr=0.02, max_epochs=100)

    # 4. MLP e RBF
    perm = np.random.permutation(N)
    s = int(0.8 * N)
    Xtr, ytr = X[perm[:s]], y[perm[:s]]
    Xte, yte = X[perm[s:]], y[perm[s:]]
    # MLP
    mlp_small = MLP([p,2,1], 'tanh', lr=0.05, max_epochs=100)
    mlp_big   = MLP([p,100,1],'tanh', lr=0.01, max_epochs=200)
    mlp_small.fit(Xtr,ytr,Xte,yte)
    mlp_big.fit(Xtr,ytr,Xte,yte)
    # RBF
    rbf_few = RBFNetwork(5, lr=0.05, max_epochs=100)
    rbf_many= RBFNetwork(30,lr=0.01, max_epochs=100)
    rbf_few.fit(Xtr,ytr,Xte,yte)
    rbf_many.fit(Xtr,ytr,Xte,yte)

    # learning curves
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
    ax1.plot(mlp_small.train_hist['acc'],'b--',label="MLP2 Train")
    ax1.plot(mlp_small.val_hist['acc'],  'b-', label="MLP2 Val")
    ax1.plot(mlp_big.train_hist['acc'],  'r--',label="MLP100 Train")
    ax1.plot(mlp_big.val_hist['acc'],    'r-', label="MLP100 Val")
    ax1.set_ylim(0,1); ax1.set_title("MLP Learning Curves"); ax1.legend(); ax1.grid(True)
    ax2.plot(rbf_few.train_hist['acc'],'b--',label="RBF5 Train")
    ax2.plot(rbf_few.val_hist['acc'],  'b-', label="RBF5 Val")
    ax2.plot(rbf_many.train_hist['acc'],'r--',label="RBF30 Train")
    ax2.plot(rbf_many.val_hist['acc'],  'r-', label="RBF30 Val")
    ax2.set_ylim(0,1); ax2.set_title("RBF Learning Curves"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()

    # 5. Monte Carlo R=250
    R = 250
    models = {"Perceptron":percep, "MLP":mlp_tanh, "RBF":rbf_net}
    results = {m:{"acc":[], "sens":[], "spec":[]} for m in models}
    for r in range(R):
        idx = np.random.permutation(N)
        Xr, yr = X[idx], y[idx]
        Xt, yt = Xr[:s], yr[:s]
        Xv, yv = Xr[s:], yr[s:]
        for name,mod in models.items():
            mod.fit(Xt,yt)
            yp = mod.predict(Xv)
            tn,fp,fn,tp = confusion_matrix_manual(yv,yp).ravel()
            results[name]["acc"].append((tp+tn)/(tp+tn+fp+fn))
            results[name]["sens"].append(tp/(tp+fn) if tp+fn>0 else 0)
            results[name]["spec"].append(tn/(tn+fp) if tn+fp>0 else 0)

    # 6. Melhor e Pior rodada
    mean_acc = np.mean([results[m]["acc"] for m in models],axis=0)
    best_i, worst_i = mean_acc.argmax(), mean_acc.argmin()
    for label, idx_case in [("Melhor",best_i),("Pior",worst_i)]:
        idx = np.random.RandomState(idx_case).permutation(N)
        Xr, yr = X[idx], y[idx]
        Xt, yt = Xr[:s], yr[:s]
        Xv, yv = Xr[s:], yr[s:]
        fig, axes = plt.subplots(1,3,figsize=(15,4))
        for ax,(name,mod) in zip(axes,models.items()):
            mod.fit(Xt,yt)
            cm = confusion_matrix_manual(yv, mod.predict(Xv))
            sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False)
            ax.set_title(name)
        fig.suptitle(f"Matrizes de Confusão ({label})", y=1.05)
        plt.tight_layout(); plt.show()
        plt.figure(figsize=(6,4))
        for name,mod in models.items():
            plt.plot(mod.train_hist['acc'],'--',label=f"{name} Train")
            plt.plot(mod.val_hist['acc'],   '-',label=f"{name} Val")
        plt.title(f"Curvas de Aprendizado ({label})")
        plt.xlabel("Época"); plt.ylabel("Acurácia")
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

    # 7. Estatísticas finais
    for met,title in [("acc","Acurácia"),("sens","Sensibilidade"),("spec","Especificidade")]:
        vals = np.array([results[n][met] for n in models])
        stats = [vals.mean(1), vals.std(1), vals.max(1), vals.min(1)]
        fig,ax = plt.subplots(); ax.axis("off")
        ax.table(cellText=np.vstack(stats).T,
                 rowLabels=list(models),
                 colLabels=["Média","Desv","Máx","Min"],
                 loc="center")
        ax.set_title(f"Estatísticas de {title}"); plt.show()
        plt.figure(figsize=(6,4))
        sns.boxplot(data=[results[n][met] for n in models], palette="Set2")
        plt.xticks(range(len(models)), list(models))
        plt.title(f"Distribuição de {title}"); plt.ylabel(title)
        plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
