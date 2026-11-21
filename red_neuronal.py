"""
Perceptrón multicapa (MLP) implementado desde cero con NumPy.

Incluye:
  * Capas densas con activación ReLU e inicialización He.
  * Salida sigmoide + entropía cruzada binaria.
  * Retropropagación (backpropagation) explícita.
  * Optimizador Adam, regularización L2 y Dropout (inverted dropout).
  * Entrenamiento por mini-lotes con parada temprana (early stopping).
"""
import numpy as np


def sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    return np.maximum(0.0, z)


def entropia_cruzada(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


class RedNeuronal:
    def __init__(self, n_entradas, capas_ocultas=(64, 32), tasa_aprendizaje=1e-3,
                 l2=1e-4, dropout=0.2, semilla=0):
        self.rng = np.random.default_rng(semilla)
        self.lr, self.l2, self.dropout = tasa_aprendizaje, l2, dropout
        tamanos = [n_entradas, *capas_ocultas, 1]
        self.W, self.b = [], []
        for n_in, n_out in zip(tamanos[:-1], tamanos[1:]):
            # Inicialización He: adecuada para ReLU.
            self.W.append(self.rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        # Momentos de Adam
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.historial = {"perdida_train": [], "perdida_val": [], "acc_train": [], "acc_val": []}

    # ------------------------------------------------------------------ forward
    def _propagar(self, X, entrenando=False):
        activaciones, mascaras, zs = [X], [], []
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = relu(z)
            if entrenando and self.dropout > 0:
                m = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                a = a * m
            else:
                m = None
            zs.append(z); mascaras.append(m); activaciones.append(a)
        salida = sigmoide(a @ self.W[-1] + self.b[-1])
        return salida, activaciones, zs, mascaras

    # ----------------------------------------------------------------- backward
    def _retropropagar(self, y, salida, activaciones, zs, mascaras):
        n = y.shape[0]
        # Derivada de BCE respecto al logit cuando la salida es sigmoide: p - y
        delta = (salida - y.reshape(-1, 1)) / n
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = activaciones[i].T @ delta + self.l2 * self.W[i]
            gb[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.W[i].T
                if mascaras[i - 1] is not None:
                    delta = delta * mascaras[i - 1]
                delta = delta * (zs[i - 1] > 0)      # derivada de ReLU
        return gW, gb

    def _adam(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i in range(len(self.W)):
            for p, g, m, v in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
                m[i] = beta1 * m[i] + (1 - beta1) * g[i]
                v[i] = beta2 * v[i] + (1 - beta2) * g[i] ** 2
                m_hat = m[i] / (1 - beta1 ** self.t)
                v_hat = v[i] / (1 - beta2 ** self.t)
                p[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # -------------------------------------------------------------- entrenar
    def entrenar(self, X, y, X_val=None, y_val=None, epocas=300, tam_lote=64,
                 paciencia=30, verbose=False):
        mejor_perdida, mejor_pesos, sin_mejora = np.inf, None, 0
        for epoca in range(1, epocas + 1):
            idx = self.rng.permutation(len(X))
            for ini in range(0, len(X), tam_lote):
                lote = idx[ini:ini + tam_lote]
                sal, act, zs, masc = self._propagar(X[lote], entrenando=True)
                gW, gb = self._retropropagar(y[lote], sal, act, zs, masc)
                self._adam(gW, gb)

            p_tr = self.predecir_proba(X)
            self.historial["perdida_train"].append(entropia_cruzada(y, p_tr))
            self.historial["acc_train"].append(np.mean((p_tr > 0.5) == y))

            if X_val is not None:
                p_val = self.predecir_proba(X_val)
                perdida_val = entropia_cruzada(y_val, p_val)
                self.historial["perdida_val"].append(perdida_val)
                self.historial["acc_val"].append(np.mean((p_val > 0.5) == y_val))
                if perdida_val < mejor_perdida - 1e-5:
                    mejor_perdida, sin_mejora = perdida_val, 0
                    mejor_pesos = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                else:
                    sin_mejora += 1
                if verbose and epoca % 10 == 0:
                    print(f"  época {epoca:3d}  pérdida_val={perdida_val:.4f}  "
                          f"acc_val={self.historial['acc_val'][-1]:.4f}")
                if sin_mejora >= paciencia:
                    break
        if mejor_pesos is not None:
            self.W, self.b = mejor_pesos
        self.epoca_final = epoca
        return self

    def predecir_proba(self, X):
        return self._propagar(X, entrenando=False)[0].ravel()

    def predecir(self, X, umbral=0.5):
        return self.predecir_proba(X) > umbral
