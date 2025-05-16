# MULTICLASSES: 1 QUESTÃO

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("coluna_vertebral.csv")

X_raw = df.iloc[:, :6].values 
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_raw) 
X_norm_T = X_norm.T  
bias = np.ones((1, X_norm_T.shape[1])) 
X = np.vstack((bias, X_norm_T)) 

def encode_label(label):
    if label == "NO":
        return [1, -1, -1]
    elif label == "DH":
        return [-1, 1, -1]
    elif label == "SL":
        return [-1, -1, 1]
    else:
        raise ValueError(f"Rótulo desconhecido: {label}")

labels = df.iloc[:, -1].values 
Y = np.array([encode_label(lbl) for lbl in labels]).T  

label_mapping = {'NO': 0, 'DH': 1, 'SL': 2}
labels = np.array([label_mapping[lbl] for lbl in df.iloc[:, -1].values])