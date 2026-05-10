"""Regla de Oja para extraer la primer componente principal iterativamente.

Implementacion numpy pura siguiendo el algoritmo de la teoria (OjaSanger.md):

    input: X (m × n) con media 0; eta; n_epochs; seed
    w ← uniforme(0, 1) en R^n (opcionalmente normalizada a norma 1)
    for epoch in 1..n_epochs:
        permutar orden de las filas
        for i in 1..m:
            y = w . x_i
            w = w + eta * y * (x_i - y * w)
        registrar w al final de cada epoca

Resultado: w_final que aproxima al primer autovector de la matriz de correlacion
de X (a menos de signo).
"""

import numpy as np


class Oja:
    """Perceptron lineal de una neurona entrenado con la regla de Oja.

    Atributos tras `fit`:
        w_:          ndarray (n,) vector de pesos final (≈ primer componente principal).
        w_history_:  ndarray (epochs + 1, n) con w al inicio y al final de cada epoca.
        history_:    list[dict] con metricas por epoca:
                        epoch, norm (||w||), cos_sim_prev, cos_sim_ref (si se paso reference).
    """

    def __init__(self, eta=1e-3, epochs=200, seed=42, init="uniform",
                 init_unit_norm=True):
        self.eta = float(eta)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.init = init
        self.init_unit_norm = bool(init_unit_norm)

    def _init_w(self, n_features, rng):
        if self.init == "uniform":
            w = rng.uniform(0.0, 1.0, size=n_features)
        elif self.init == "gaussian":
            w = rng.standard_normal(n_features)
        else:
            raise ValueError(f"init desconocido: {self.init}")
        if self.init_unit_norm:
            w = w / (np.linalg.norm(w) + 1e-12)
        return w

    @staticmethod
    def _cos_sim(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    def fit(self, X, reference=None):
        """Entrena Oja sobre X (centrada o estandarizada).

        Args:
            X: ndarray (m, n). Asume media 0 por columna.
            reference: ndarray (n,) opcional. Si se pasa, se registra
                       cos_sim(w_t, reference) por epoca (sirve para trackear
                       convergencia hacia la PC1 calculada con libreria).
        """
        X = np.asarray(X, dtype=np.float64)
        m, n = X.shape
        rng = np.random.default_rng(self.seed)

        w = self._init_w(n, rng)
        w_history = [w.copy()]
        history = [{
            "epoch": 0,
            "norm": float(np.linalg.norm(w)),
            "cos_sim_prev": 1.0,
            "cos_sim_ref": (self._cos_sim(w, reference) if reference is not None else None),
        }]

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(m)
            w_prev = w.copy()
            for i in order:
                x = X[i]
                y = float(np.dot(w, x))
                w = w + self.eta * y * (x - y * w)
            w_history.append(w.copy())
            history.append({
                "epoch": epoch,
                "norm": float(np.linalg.norm(w)),
                "cos_sim_prev": self._cos_sim(w, w_prev),
                "cos_sim_ref": (self._cos_sim(w, reference) if reference is not None else None),
            })

        self.w_ = w
        self.w_history_ = np.asarray(w_history)
        self.history_ = history
        return self

    def transform(self, X):
        """Proyecta X sobre la PC1 aprendida. Devuelve (m,) con los scores."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_
