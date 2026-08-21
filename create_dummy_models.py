import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.mean_ = np.zeros(1280)
scaler.scale_ = np.ones(1280)
scaler.var_ = np.ones(1280)

class RRELM:
    def __init__(self):
        self.W = np.random.randn(1280, 1024)
        self.b = np.random.randn(1024)
        self.beta = np.random.randn(1024, 4)
    def _activate(self, H):
        return 1.0 / (1.0 + np.exp(-H))
    def predict_proba(self, X):
        H = self._activate(X @ self.W + self.b)
        return H @ self.beta

rrelm = RRELM()

with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('model/rrelm.pkl', 'wb') as f:
    pickle.dump(rrelm, f)
