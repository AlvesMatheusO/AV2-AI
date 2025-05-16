# MULTICLASSES: 6 QUESTÃO 

import numpy as np

accs_adaline = np.load('accs_adaline.npy')
accs_mlp = np.load('accs_mlp.npy')
accs_rbf = np.load('accs_rbf.npy')

def print_model_results(model_name, accs):
    print(f"{model_name}:")
    print(f"  Média         = {np.mean(accs):.4f}")
    print(f"  Desvio-Padrão = {np.std(accs):.4f}")
    print(f"  Maior Valor   = {np.max(accs):.4f}")
    print(f"  Menor Valor   = {np.min(accs):.4f}")
    print()

print_model_results("ADALINE", accs_adaline)
print_model_results("MLP", accs_mlp)
print_model_results("RBF", accs_rbf)
