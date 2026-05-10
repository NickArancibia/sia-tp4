"""PCA con libreria.

Camino principal: sklearn.decomposition.PCA sobre datos estandarizados.
Camino de validacion cruzada: numpy.linalg.eigh sobre la matriz de correlacion.
Se exponen ambos resultados para que el bloque de analisis pueda chequear que
coinciden (a menos de signo) y reportar cosine similarity por componente.
"""

import numpy as np
from sklearn.decomposition import PCA as SKPCA


def _normalize_signs(V):
    """Convencion de signo: en cada columna, fuerza que la componente de mayor
    valor absoluto sea positiva. Esto elimina la ambiguedad de signo de los
    autovectores y vuelve comparables los outputs de sklearn y eigh.

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
            eigenvalues:               (n,) descendente, varianza explicada absoluta (sklearn).
            eigenvectors:              (n, n) columnas = componentes principales (sign-normalized).
            loadings:                  alias de eigenvectors. Filas = features, columnas = PCs.
            explained_variance_ratio:  (n,) proporcion de varianza por componente.
            explained_variance_cum:    (n,) acumulada.
            scores:                    (m, n) = X_std @ eigenvectors. Proyeccion de los datos.
            eigh_eigenvalues:          (n,) descendente, autovalores de np.corrcoef(X.T).
            eigh_eigenvectors:         (n, n) autovectores de corrcoef, sign-normalized.
            cross_check_cos_sim:       (n,) cosine similarity en valor absoluto entre cada
                                       PC sklearn y el autovector correspondiente de eigh.
                                       Sanity check: deberia ser ~1 para todas.
    """
    X_std = np.asarray(X_std, dtype=np.float64)
    n = X_std.shape[1]

    sk = SKPCA(n_components=n, svd_solver="full")
    sk.fit(X_std)
    skl_V = _normalize_signs(sk.components_.T)

    R = np.corrcoef(X_std.T)
    eig_vals, eig_vecs = np.linalg.eigh(R)
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = _normalize_signs(eig_vecs[:, order])

    cos_sim = np.array([
        abs(float(np.dot(skl_V[:, k], eig_vecs[:, k])))
        for k in range(n)
    ])

    scores = X_std @ skl_V

    return {
        "eigenvalues": sk.explained_variance_,
        "eigenvectors": skl_V,
        "loadings": skl_V,
        "explained_variance_ratio": sk.explained_variance_ratio_,
        "explained_variance_cum": np.cumsum(sk.explained_variance_ratio_),
        "scores": scores,
        "eigh_eigenvalues": eig_vals,
        "eigh_eigenvectors": eig_vecs,
        "cross_check_cos_sim": cos_sim,
    }
