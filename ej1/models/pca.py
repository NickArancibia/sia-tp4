"""PCA con libreria (sklearn).

Utiliza sklearn.decomposition.PCA sobre datos estandarizados.
Los autovectores se normalizan en signo para que sean comparables
con la PC1 obtenida por Oja.
"""

import numpy as np
from sklearn.decomposition import PCA as SKPCA


def _normalize_signs(V):
    """Convencion de signo: en cada columna, fuerza que la componente de mayor
    valor absoluto sea positiva. Esto elimina la ambiguedad de signo de los
    autovectores y vuelve comparables los outputs de sklearn y Oja.

    V: ndarray (n_features, n_components). Devuelve una copia.
    """
    V = np.array(V, copy=True)
    for k in range(V.shape[1]):
        idx = int(np.argmax(np.abs(V[:, k])))
        if V[idx, k] < 0:
            V[:, k] *= -1
    return V


def run_pca(X_std):
    """Calcula PCA sobre datos ya estandarizados.

    Args:
        X_std: ndarray (m, n) con media 0 y std 1 por columna.

    Returns:
        dict con:
            eigenvalues:               (n,) descendente, varianza explicada absoluta.
            eigenvectors:              (n, n) columnas = componentes principales (sign-normalized).
            loadings:                  alias de eigenvectors. Filas = features, columnas = PCs.
            explained_variance_ratio:  (n,) proporcion de varianza por componente.
            explained_variance_cum:    (n,) acumulada.
            scores:                    (m, n) = X_std @ eigenvectors. Proyeccion de los datos.
    """
    X_std = np.asarray(X_std, dtype=np.float64)
    n = X_std.shape[1]

    sk = SKPCA(n_components=n, svd_solver="full")
    sk.fit(X_std)
    skl_V = _normalize_signs(sk.components_.T)

    scores = X_std @ skl_V

    return {
        "eigenvalues": sk.explained_variance_,
        "eigenvectors": skl_V,
        "loadings": skl_V,
        "explained_variance_ratio": sk.explained_variance_ratio_,
        "explained_variance_cum": np.cumsum(sk.explained_variance_ratio_),
        "scores": scores,
    }
