"""Analisis y visualizacion del modelo de Oja.

Las funciones consumen los atributos `w_`, `w_history_`, `history_` de una
instancia entrenada de `models.oja.Oja` y los datos auxiliares (feature names,
PC1 de referencia obtenida con libreria, etc.).
"""

import numpy as np
import matplotlib.pyplot as plt

from shared.plotting import save_fig, diverging_colors, PALETTE


def align_sign(w, reference):
    """Devuelve w con signo ajustado para que cos_sim(w, reference) > 0.
    Si reference es None, devuelve w sin cambios.
    """
    if reference is None:
        return w
    if np.dot(w, reference) < 0:
        return -w
    return w


def spearman_corr(a, b):
    """Coeficiente de Spearman entre dos vectores 1D (correlacion de los rangos)."""
    a = np.asarray(a)
    b = np.asarray(b)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def comparison_metrics(w_oja, pc1_ref, scores_oja, scores_ref):
    """Devuelve un dict con metricas de comparacion Oja vs PCA libreria.

    Asume que w_oja ya esta alineado en signo con pc1_ref.
    """
    cos_sim = float(np.dot(w_oja, pc1_ref) /
                    (np.linalg.norm(w_oja) * np.linalg.norm(pc1_ref) + 1e-12))
    max_abs_err = float(np.abs(w_oja - pc1_ref).max())
    rmse = float(np.sqrt(np.mean((w_oja - pc1_ref) ** 2)))
    pearson = float(np.corrcoef(scores_oja, scores_ref)[0, 1])
    spearman = spearman_corr(scores_oja, scores_ref)
    return {
        "cos_sim": cos_sim,
        "max_abs_error_loadings": max_abs_err,
        "rmse_loadings": rmse,
        "pearson_scores": pearson,
        "spearman_ranking": spearman,
        "norm_oja": float(np.linalg.norm(w_oja)),
    }


def plot_convergence(history, path):
    """Dos paneles: ||w|| por epoca + cos_sim(w_t, PC1 libreria) por epoca.
    Si la historia no tiene cos_sim_ref (entrenamiento sin reference), omite el panel.
    """
    epochs = [h["epoch"] for h in history]
    norms = [h["norm"] for h in history]
    has_ref = history[-1]["cos_sim_ref"] is not None
    cos_ref = [h["cos_sim_ref"] for h in history] if has_ref else None

    n_panels = 2 if has_ref else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.5 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(epochs, norms, color=PALETTE["primary"], linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, label="||w|| = 1")
    ax.set_ylabel("||w||")
    ax.set_title("Convergencia de la norma del vector de pesos")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    if has_ref:
        ax = axes[1]
        # Usar valor absoluto para que sea robusto al signo
        ax.plot(epochs, np.abs(cos_ref), color=PALETTE["highlight"], linewidth=1.5,
                label="|cos_sim(w_t, PC1_libreria)|")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(0, 1.05)
        ax.set_title("Convergencia hacia la PC1 de la libreria")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    else:
        axes[0].set_xlabel("Epoca")

    fig.tight_layout()
    save_fig(fig, path)


def plot_weights_evolution(w_history, feature_names, path, reference=None):
    """Evolucion de cada componente de w a lo largo de las epocas (una linea por feature).
    Si se pasa `reference` (PC1 sklearn alineada), dibuja una linea horizontal
    punteada para cada feature con su valor de referencia.
    """
    w_history = np.asarray(w_history)
    n_epochs_plus1, n_feat = w_history.shape
    epochs = np.arange(n_epochs_plus1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_feat))
    for j, name in enumerate(feature_names):
        ax.plot(epochs, w_history[:, j], color=colors[j], linewidth=1.6, label=name)
        if reference is not None:
            ax.axhline(reference[j], color=colors[j], linestyle=":", linewidth=0.9, alpha=0.6)

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Valor de la componente del vector de pesos")
    title = "Evolucion de w por epoca"
    if reference is not None:
        title += "  (lineas punteadas: PC1 libreria)"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def plot_comparison_loadings(w_oja, pc1_ref, feature_names, path):
    """Barras pareadas por feature: cargas Oja vs cargas sklearn."""
    n = len(feature_names)
    x = np.arange(n)
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, w_oja, width, color=PALETTE["primary"],
                edgecolor="black", linewidth=0.4, label="Oja")
    b2 = ax.bar(x + width / 2, pc1_ref, width, color=PALETTE["highlight"],
                edgecolor="black", linewidth=0.4, label="Libreria (sklearn)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("Carga (loading)")
    ax.set_title("Cargas de PC1: Oja vs Libreria")
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + (0.01 if h >= 0 else -0.03),
                    f"{h:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)


def plot_comparison_scores(scores_oja, scores_ref, countries, path):
    """Scatter de scores PC1 por pais: x = sklearn, y = Oja. Linea y=x de referencia.
    Anota los puntos extremos con el nombre del pais.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(scores_ref, scores_oja, s=40, color=PALETTE["primary"], alpha=0.8,
               edgecolor="navy", zorder=3)
    lo = float(min(scores_ref.min(), scores_oja.min()))
    hi = float(max(scores_ref.max(), scores_oja.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=0.8,
            label="y = x")

    # Anotar los 4 extremos
    idx_top = np.argsort(scores_ref)[::-1][:3]
    idx_bot = np.argsort(scores_ref)[:3]
    for i in np.concatenate([idx_top, idx_bot]):
        ax.annotate(countries[i], (scores_ref[i], scores_oja[i]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("Score PC1 (libreria)")
    ax.set_ylabel("Score PC1 (Oja)")
    ax.set_title("Scores por pais: Oja vs Libreria")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    save_fig(fig, path)


def plot_oja_ranking(scores_oja, countries, path):
    """Ranking horizontal de paises segun la PC1 obtenida por Oja."""
    order = np.argsort(scores_oja)
    sorted_scores = scores_oja[order]
    sorted_countries = [countries[i] for i in order]
    colors = diverging_colors(sorted_scores)

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(countries) + 1))
    y = np.arange(len(sorted_countries))
    ax.barh(y, sorted_scores, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_countries, fontsize=9)
    ax.set_xlabel("PC1 (Oja)")
    ax.set_title("Ranking de paises segun PC1 obtenida por Oja")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, path)
