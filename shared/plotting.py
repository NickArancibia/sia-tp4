import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "primary": "steelblue",
    "secondary": "tomato",
    "accent": "slateblue",
    "highlight": "crimson",
    "neutral": "dimgray",
    "positive": "#2a9d8f",
    "negative": "#e76f51",
}

DEFAULT_DPI = 150


def save_fig(fig, path, dpi=DEFAULT_DPI):
    """Guarda una figura asegurando que el directorio existe y cerrándola tras escribir."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def diverging_colors(values, cmap="RdBu_r", vmin=None, vmax=None):
    """Devuelve una lista de colores RGBA mapeando `values` a una paleta divergente
    centrada en 0. Útil para barras con signo (ranking de PC1, cargas, etc.)."""
    import numpy as np
    arr = np.asarray(values, dtype=float)
    if vmin is None or vmax is None:
        absmax = float(max(abs(arr.min()), abs(arr.max()), 1e-12))
        vmin = -absmax if vmin is None else vmin
        vmax = absmax if vmax is None else vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    mapper = plt.get_cmap(cmap)
    return [mapper(norm(v)) for v in arr]
