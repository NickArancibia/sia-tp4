"""Analisis y visualizacion del SOM (red de Kohonen).

Genera los graficos que pide la consigna 1.1:
- Asociacion de paises a neuronas (mapa de paises).
- Distancias promedio entre neuronas vecinas (matriz U).
- Cantidad de elementos asociados a cada neurona (mapa de hits).
- Mapas por variable (interpretacion de cada region).
- Convergencia del error de cuantizacion.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE


def assign_countries(som, X, countries):
    """Devuelve dict {(i, j): [country, ...]} con los paises asignados a cada neurona."""
    assignments = {}
    for x, name in zip(X, countries):
        i, j = som._winner(x, som.W)
        assignments.setdefault((i, j), []).append(name)
    return assignments


def plot_country_map(som, X, countries, path):
    """Grilla k x k. Cada celda muestra los paises asignados a esa neurona.
    Fondo coloreado por el conteo de hits para destacar las celdas pobladas.
    """
    k = som.grid_size
    assignments = assign_countries(som, X, countries)
    hits = som.hits(X)

    fig, ax = plt.subplots(figsize=(max(8, 2 * k), max(7, 1.8 * k)))
    im = ax.imshow(hits, cmap="Blues", vmin=0, vmax=max(1, hits.max()))
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"col {j}" for j in range(k)])
    ax.set_yticklabels([f"fila {i}" for i in range(k)])
    ax.set_xlabel("Columna de la grilla")
    ax.set_ylabel("Fila de la grilla")
    ax.set_title("Mapa de paises asignados a cada neurona")

    for (i, j), names in assignments.items():
        text = "\n".join(names)
        ax.text(j, i, text, ha="center", va="center",
                fontsize=8.5, color="black" if hits[i, j] < hits.max() * 0.6 else "white",
                fontweight="medium")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cantidad de paises")
    fig.tight_layout()
    save_fig(fig, path)


def plot_u_matrix(som, path):
    """Heatmap de la matriz U: distancia promedio a vecinos.
    Tonos claros = neuronas similares (clusters). Tonos oscuros = fronteras.
    """
    U = som.u_matrix()
    k = som.grid_size

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * k), max(5, 1.4 * k)))
    im = ax.imshow(U, cmap="bone_r")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"col {j}" for j in range(k)])
    ax.set_yticklabels([f"fila {i}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{U[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if U[i, j] < U.max() * 0.6 else "white")
    ax.set_title("Matriz U - distancia promedio entre neuronas vecinas")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distancia promedio")
    fig.tight_layout()
    save_fig(fig, path)


def plot_hits(som, X, path):
    """Heatmap del conteo de paises por neurona."""
    H = som.hits(X)
    k = som.grid_size

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * k), max(5, 1.4 * k)))
    im = ax.imshow(H, cmap="Blues", vmin=0, vmax=max(1, H.max()))
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"col {j}" for j in range(k)])
    ax.set_yticklabels([f"fila {i}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(H[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="black" if H[i, j] < H.max() * 0.6 else "white")
    ax.set_title("Cantidad de paises asignados a cada neurona")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cantidad")
    fig.tight_layout()
    save_fig(fig, path)


def plot_variable_maps(som, feature_names, path):
    """Una grilla de subplots, uno por variable, mostrando el valor del peso
    correspondiente en cada neurona. Permite interpretar que representa
    cada region del mapa.
    """
    n_feat = len(feature_names)
    k = som.grid_size
    cols = min(3, n_feat)
    rows = int(np.ceil(n_feat / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows))
    axes = np.atleast_2d(axes).flatten()

    for idx, name in enumerate(feature_names):
        ax = axes[idx]
        layer = som.W[:, :, idx]
        vmax = float(np.abs(layer).max())
        im = ax.imshow(layer, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([str(j) for j in range(k)], fontsize=10)
        ax.set_yticklabels([str(i) for i in range(k)], fontsize=10)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{layer[i, j]:+.2f}", ha="center", va="center",
                        fontsize=10,
                        color="black" if abs(layer[i, j]) < 0.7 * vmax else "white",
                        fontweight="medium")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n_feat, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Mapas por variable - valor del peso por neurona",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, path)


def plot_convergence(history, path):
    """Error de cuantizacion (QE) por epoca: distancia media de cada muestra
    a su neurona ganadora. Deberia decrecer monotonicamente.
    Incluye un segundo eje con eta y radius.
    """
    epochs = [h["epoch"] for h in history]
    qes = [h["qe"] for h in history]
    etas = [h["eta"] for h in history]
    radii = [h["radius"] for h in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, qes, color=PALETTE["primary"], linewidth=1.6,
             label="QE (distancia media a la ganadora)")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Quantization Error", color=PALETTE["primary"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["primary"])
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, etas, color=PALETTE["highlight"], linewidth=1.0,
             linestyle="--", label="eta(t)")
    ax2.plot(epochs, radii, color=PALETTE["accent"], linewidth=1.0,
             linestyle=":", label="radius(t)")
    ax2.set_ylabel("eta / radius")
    ax2.tick_params(axis="y")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    ax1.set_title("Convergencia del SOM")
    fig.tight_layout()
    save_fig(fig, path)
