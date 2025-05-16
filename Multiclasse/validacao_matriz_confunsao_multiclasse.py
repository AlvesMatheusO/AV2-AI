# MULTICLASSES: 5 QUESTÃO


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from adaline_multiclasse import Adaline
from mlp_multiclasse import MLP
from rbf_multiclasse import RBF
from organizacao_X_Y_coluna import X, labels

def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def run_model_and_store_predictions(model_class, model_name, model_params, X, labels, is_adaline=False):
    R = 100
    accs = []
    predictions = []
    true_labels = []

    for _ in range(R):
        X_train, X_test, y_train_idx, y_test_idx = train_test_split(
            X.T, labels, test_size=0.2, stratify=labels
        )
        X_train = X_train.T
        X_test = X_test.T

        Y_train_bipolar = np.array([
            [1, -1, -1] if lbl == 0 else
            [-1, 1, -1] if lbl == 1 else
            [-1, -1, 1] for lbl in y_train_idx
        ]).T

        if is_adaline:
            models = []
            for i in range(Y_train_bipolar.shape[0]):
                m = model_class(**model_params)
                m.train(X_train, Y_train_bipolar[i])
                models.append(m)
            outputs = [m.w @ X_test for m in models]
            outputs = np.array(outputs)
            y_pred = np.argmax(outputs, axis=0)
        else:
            model = model_class(**model_params)
            model.train(X_train, Y_train_bipolar)
            y_pred = model.predict(X_test)

        acc = np.mean(y_pred == y_test_idx)
        accs.append(acc)
        predictions.append(y_pred)
        true_labels.append(y_test_idx)

    accs = np.array(accs)
    best_idx = accs.argmax()
    worst_idx = accs.argmin()

    print(f"== {model_name} ==")
    print(f"Melhor acurácia: {accs[best_idx]:.4f}")
    print(f"Pior acurácia: {accs[worst_idx]:.4f}")

    for label, idx_case in [("Melhor", best_idx), ("Pior", worst_idx)]:
        cm = confusion_matrix(true_labels[idx_case], predictions[idx_case], n_classes=3)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"{model_name} - Matriz de Confusão ({label} Caso)")
        plt.xlabel("Predito")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

# === EXECUÇÃO PARA OS 3 MODELOS ===

run_model_and_store_predictions(
    model_class=Adaline,
    model_name="ADALINE",
    model_params={"input_dim": 7, "learning_rate": 0.01, "max_epochs": 1000},
    X=X,
    labels=labels,
    is_adaline=True
)

run_model_and_store_predictions(
    model_class=MLP,
    model_name="MLP",
    model_params={"input_dim": 7, "hidden_dim": 10, "output_dim": 3, "learning_rate": 0.01, "max_epochs": 1000},
    X=X,
    labels=labels
)

run_model_and_store_predictions(
    model_class=RBF,
    model_name="RBF",
    model_params={"input_dim": 7, "hidden_dim": 10, "output_dim": 3, "sigma": 1.0},
    X=X,
    labels=labels
)
