import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mlp_multiclasse import MLP
from organizacao_X_Y_coluna import X, labels

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def sensitivity_specificity(conf_mat):
    sens = []
    spec = []
    for i in range(conf_mat.shape[0]):
        TP = conf_mat[i, i]
        FN = np.sum(conf_mat[i, :]) - TP
        FP = np.sum(conf_mat[:, i]) - TP
        TN = np.sum(conf_mat) - (TP + FP + FN)
        sens.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        spec.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    return np.mean(sens), np.mean(spec)

def monte_carlo_mlp(X, Y_labels):
    R = 100
    accs = []
    sens_list = []
    spec_list = []
    n_classes = len(np.unique(Y_labels))

    for _ in range(R):
        X_train, X_test, y_train_idx, y_test_idx = train_test_split(
            X.T, Y_labels, test_size=0.2, stratify=Y_labels
        )
        X_train = X_train.T
        X_test = X_test.T

        Y_train_bipolar = np.array([
            [1, -1, -1] if lbl == 0 else
            [-1, 1, -1] if lbl == 1 else
            [-1, -1, 1] for lbl in y_train_idx
        ]).T

        model = MLP(input_dim=7, hidden_dim=10, output_dim=3, learning_rate=0.01, max_epochs=1000)
        model.train(X_train, Y_train_bipolar)
        y_pred = model.predict(X_test)

        y_pred = y_pred.astype(int)
        y_test_idx = y_test_idx.astype(int)

        acc = accuracy(y_test_idx, y_pred)
        conf_mat = confusion_matrix(y_test_idx, y_pred, n_classes)
        sens, spec = sensitivity_specificity(conf_mat)

        accs.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)

    return accs, sens_list, spec_list

accs, sens, specs = monte_carlo_mlp(X, labels)

# execução
print(f"Acurácia: Média={np.mean(accs):.4f}, Desvio={np.std(accs):.4f}, Max={np.max(accs):.4f}, Min={np.min(accs):.4f}")
print(f"Sensibilidade: Média={np.mean(sens):.4f}, Desvio={np.std(sens):.4f}, Max={np.max(sens):.4f}, Min={np.min(sens):.4f}")
print(f"Especificidade: Média={np.mean(specs):.4f}, Desvio={np.std(specs):.4f}, Max={np.max(specs):.4f}, Min={np.min(specs):.4f}")

np.save('accs_mlp.npy', accs)
