"""Self-Organizing Map (red de Kohonen).

Implementacion numpy pura siguiendo la teoria (Kohonen.md):

    Grilla 2D k x k de neuronas. Cada neurona j tiene un vector de pesos W_j en R^n.
    En cada iteracion:
        1) Se toma una muestra x.
        2) Se encuentra la neurona ganadora k_hat = argmin_j ||x - W_j||.
        3) Se actualizan los pesos de todas las neuronas j cuya distancia en la
           grilla a k_hat sea <= R(t):
                W_j ← W_j + eta(t) · (x - W_j)
    eta(t) decae a eta_final y R(t) decae a radius_final linealmente.
    Inicializacion preferible: muestras aleatorias del dataset.
"""

import numpy as np


class SOM:
    """Red de Kohonen / SOM con grilla rectangular k x k.

    Tras `fit`, expone:
        W:           ndarray (k, k, n) con los pesos finales.
        history_:    list[dict] con el error de cuantizacion (QE) por epoca.
        W_init_:     pesos iniciales (copia, para diagnostico).
    """

    def __init__(self, grid_size=4, epochs=500, eta_0=0.5, eta_final=0.01,
                 radius_0=2.0, radius_final=1.0, init="samples", seed=42):
        self.grid_size = int(grid_size)
        self.epochs = int(epochs)
        self.eta_0 = float(eta_0)
        self.eta_final = float(eta_final)
        self.radius_0 = float(radius_0)
        self.radius_final = float(radius_final)
        self.init = init
        self.seed = int(seed)

    def _init_weights(self, X, rng):
        k = self.grid_size
        n = X.shape[1]
        if self.init == "samples":
            idx = rng.integers(0, X.shape[0], size=k * k)
            W = X[idx].reshape(k, k, n).astype(np.float64)
            # Pequeño jitter para que neuronas inicializadas con la misma muestra no queden identicas.
            W = W + rng.normal(0, 1e-3, size=W.shape)
            return W
        if self.init == "random":
            return rng.uniform(-1.0, 1.0, size=(k, k, n))
        raise ValueError(f"init desconocido: {self.init}")

    def _winner(self, x, W):
        """Devuelve (i, j) de la neurona con menor distancia euclidea a x."""
        diff = W - x  # broadcast (k, k, n)
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        idx = int(np.argmin(d2))
        return divmod(idx, self.grid_size)

    def _schedule_linear(self, t, T, v0, v_final):
        alpha = t / max(T - 1, 1)
        return v0 + (v_final - v0) * alpha

    def _quantization_error(self, X, W):
        """Error medio entre cada muestra y su neurona ganadora."""
        total = 0.0
        for x in X:
            i, j = self._winner(x, W)
            total += float(np.linalg.norm(x - W[i, j]))
        return total / X.shape[0]

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        m, n = X.shape
        k = self.grid_size
        rng = np.random.default_rng(self.seed)

        W = self._init_weights(X, rng)
        self.W_init_ = W.copy()

        # Coordenadas de cada neurona en la grilla (para calculo de vecindario).
        ii, jj = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")

        history = [{
            "epoch": 0,
            "eta": self.eta_0,
            "radius": self.radius_0,
            "qe": self._quantization_error(X, W),
        }]

        # T = epochs * m = total de iteraciones. Decay lineal de eta y radius en t.
        T = self.epochs * m
        t = 0
        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(m)
            for i in order:
                x = X[i]
                wi, wj = self._winner(x, W)
                eta_t = self._schedule_linear(t, T, self.eta_0, self.eta_final)
                R_t = self._schedule_linear(t, T, self.radius_0, self.radius_final)

                # Mascara del vecindario: distancia euclidea en la grilla <= R_t.
                d_grid = np.sqrt((ii - wi) ** 2 + (jj - wj) ** 2)
                mask = d_grid <= R_t  # (k, k)
                # Aplicar regla de Kohonen solo a las neuronas dentro del vecindario.
                W[mask] = W[mask] + eta_t * (x - W[mask])
                t += 1

            history.append({
                "epoch": epoch,
                "eta": float(eta_t),
                "radius": float(R_t),
                "qe": self._quantization_error(X, W),
            })

        self.W = W
        self.history_ = history
        return self

    def winners(self, X):
        """Devuelve la lista de (i, j) ganadores para cada fila de X."""
        return [self._winner(x, self.W) for x in X]

    def hits(self, X):
        """Devuelve un ndarray (k, k) con el conteo de muestras que caen en cada neurona."""
        k = self.grid_size
        H = np.zeros((k, k), dtype=int)
        for x in X:
            i, j = self._winner(x, self.W)
            H[i, j] += 1
        return H

    def u_matrix(self):
        """Matriz U: para cada neurona, distancia promedio a sus vecinos en la grilla.
        Usa vecindario 8 (Moore): incluye los 4-vecinos directos y los 4 diagonales.
        Devuelve ndarray (k, k).
        """
        k = self.grid_size
        U = np.zeros((k, k), dtype=np.float64)
        offsets = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]
        for i in range(k):
            for j in range(k):
                dists = []
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < k and 0 <= nj < k:
                        dists.append(float(np.linalg.norm(self.W[i, j] - self.W[ni, nj])))
                U[i, j] = float(np.mean(dists)) if dists else 0.0
        return U
