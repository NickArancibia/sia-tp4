"""Analisis y visualizacion del PCA con libreria.

Cada funcion consume outputs del modelo (`models.pca.run_pca`) o los datos
crudos y produce una figura. Las firmas reciben rutas absolutas o relativas;
las figuras se guardan con `shared.plotting.save_fig`.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared.plotting import save_fig, diverging_colors, PALETTE


def plot_boxplots(X, feature_names, path, title, standardized=False):
    """Boxplot por feature. Si `standardized=True` agrega una linea horizontal en 0
    y ajusta el rotulo del eje Y.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(X, tick_labels=feature_names, showmeans=False, patch_artist=True,
               boxprops=dict(facecolor=PALETTE["primary"], alpha=0.5,
                             edgecolor=PALETTE["primary"]),
               medianprops=dict(color=PALETTE["highlight"], linewidth=1.5),
               whiskerprops=dict(color=PALETTE["primary"]),
               capprops=dict(color=PALETTE["primary"]),
               flierprops=dict(marker="o", markerfacecolor=PALETTE["secondary"],
                               markersize=4, alpha=0.6))
    ax.set_title(title)
    ax.set_ylabel("Valor estandarizado" if standardized else "Valor original")
    ax.tick_params(axis="x", rotation=20)
    if standardized:
        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def plot_correlation_heatmap(X, feature_names, path):
    """Heatmap de la matriz de correlacion (Pearson) entre features.
    Anota cada celda con el valor numerico.
    """
    R = np.corrcoef(X.T)
    n = len(feature_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(R, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=35, ha="right")
    ax.set_yticklabels(feature_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(R[i, j]) < 0.6 else "white")
    ax.set_title("Matriz de correlacion entre variables")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, path)


def plot_scree(explained_variance_ratio, path, threshold=0.8):
    """Scree plot: barras de varianza explicada por PC + linea acumulada.
    Marca la linea horizontal del umbral acumulado pedido (e.g. 0.8) y la
    componente a partir de la cual se cruza.
    """
    evr = np.asarray(explained_variance_ratio)
    cum = np.cumsum(evr)
    n = len(evr)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(x, evr, color=PALETTE["primary"], alpha=0.75,
                   label="Varianza explicada")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Proporcion de varianza explicada")
    ax1.set_xticks(x)
    ax1.set_ylim(0, max(1.0, evr.max() * 1.15))
    for b, v in zip(bars, evr):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x, cum, color=PALETTE["highlight"], marker="o", linewidth=1.8,
             label="Varianza acumulada")
    ax2.set_ylabel("Varianza acumulada", color=PALETTE["highlight"])
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor=PALETTE["highlight"])
    ax2.axhline(threshold, color="gray", linestyle="--", linewidth=0.8,
                label=f"Umbral {int(threshold * 100)}%")
    cross = int(np.argmax(cum >= threshold)) + 1 if cum.max() >= threshold else n
    ax2.axvline(cross, color="gray", linestyle=":", linewidth=0.8)
    for xi, c in zip(x, cum):
        ax2.text(xi, c + 0.025, f"{c:.2f}", ha="center", fontsize=8,
                 color=PALETTE["highlight"])

    ax1.set_title("Scree plot - Varianza explicada por componente")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    save_fig(fig, path)


def plot_loadings_heatmap(loadings, feature_names, path):
    """Heatmap de cargas (features x PCs).
    loadings: ndarray (n_features, n_components).
    """
    n_feat, n_pc = loadings.shape
    fig, ax = plt.subplots(figsize=(max(6, n_pc * 0.8 + 4), max(4, n_feat * 0.5 + 1)))
    vmax = np.abs(loadings).max()
    im = ax.imshow(loadings, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(feature_names)
    ax.set_xticks(range(n_pc))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_pc)])
    for i in range(n_feat):
        for j in range(n_pc):
            ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(loadings[i, j]) < 0.5 else "white")
    ax.set_title("Cargas (loadings) de cada variable en cada componente principal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, path)


def plot_pc1_pc2_loadings_bars(loadings, feature_names, path):
    """Barras agrupadas por variable: compara las cargas de PC1 y PC2."""
    vals = np.asarray(loadings, dtype=float)
    pc1 = vals[:, 0]
    pc2 = vals[:, 1]
    x = np.arange(len(feature_names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars1 = ax.bar(x - width / 2, pc1, width, color=PALETTE["primary"],
                   edgecolor="black", linewidth=0.4, label="PC1")
    bars2 = ax.bar(x + width / 2, pc2, width, color=PALETTE["highlight"],
                   edgecolor="black", linewidth=0.4, label="PC2")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("Carga")
    ax.set_title("Cargas por variable en PC1 y PC2")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + (0.018 if height >= 0 else -0.035),
                    f"{height:+.2f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8)

    fig.tight_layout()
    save_fig(fig, path)


def plot_biplot(scores, loadings, countries, feature_names, path,
                pc_x=0, pc_y=1, scale_arrows=None):
    """Biplot PC_x vs PC_y: scatter de paises + flechas de cargas.

    scores: (m, n) coordenadas de cada pais en cada PC.
    loadings: (n_features, n) cargas.
    pc_x, pc_y: indices (0-based) de las componentes a graficar.
    scale_arrows: factor de escalado de las flechas. Si None, se elige automatico
                  para que las flechas ocupen un rango comparable a los scores.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xs = scores[:, pc_x]
    ys = scores[:, pc_y]
    ax.scatter(xs, ys, s=40, color=PALETTE["primary"], alpha=0.7,
               edgecolor="navy", zorder=3)
    for i, name in enumerate(countries):
        ax.annotate(name, (xs[i], ys[i]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points", zorder=4)

    score_range = float(max(np.ptp(xs), np.ptp(ys)))
    if scale_arrows is None:
        load_max = np.abs(loadings[:, [pc_x, pc_y]]).max()
        scale_arrows = score_range / (load_max * 2.2 + 1e-9)

    for j, fname in enumerate(feature_names):
        dx = loadings[j, pc_x] * scale_arrows
        dy = loadings[j, pc_y] * scale_arrows
        ax.arrow(0, 0, dx, dy, color=PALETTE["highlight"], alpha=0.85,
                 width=0.005 * score_range if scale_arrows else 0.005,
                 head_width=0.025 * score_range, length_includes_head=True,
                 zorder=2)
        ax.text(dx * 1.08, dy * 1.08, fname, color=PALETTE["highlight"],
                fontsize=10, fontweight="bold", ha="center", va="center", zorder=5)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.set_xlabel(f"PC{pc_x + 1}")
    ax.set_ylabel(f"PC{pc_y + 1}")
    ax.set_title(f"Biplot PC{pc_x + 1} vs PC{pc_y + 1}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_fig(fig, path)


def plot_pc_ranking(scores_pc, countries, path, pc_index=0):
    """Ranking horizontal de paises segun una componente principal.
    Colorea positivos vs negativos para enfatizar el signo del indice.
    """
    order = np.argsort(scores_pc)
    sorted_scores = scores_pc[order]
    sorted_countries = [countries[i] for i in order]
    colors = diverging_colors(sorted_scores)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(countries) + 1))
    y = np.arange(len(sorted_countries))
    ax.barh(y, sorted_scores, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_countries, fontsize=9)
    ax.set_xlabel(f"PC{pc_index + 1}")
    ax.set_title(f"Ranking de paises segun PC{pc_index + 1}")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def summarize_pc1_interpretation(loadings, feature_names, pc_index=0):
    """Devuelve un dict con la interpretacion textual de la PC.
    Util para imprimir en stdout y para incluir en el README/slides.
    """
    pc = loadings[:, pc_index]
    items = sorted(
        [(name, float(val)) for name, val in zip(feature_names, pc)],
        key=lambda t: -abs(t[1]),
    )
    return {
        "pc_index": pc_index,
        "loadings_sorted_by_abs": items,
        "positive": [(n, v) for n, v in items if v > 0],
        "negative": [(n, v) for n, v in items if v < 0],
    }
