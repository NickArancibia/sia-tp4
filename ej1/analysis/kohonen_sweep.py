"""Analisis comparativo del SOM bajo distintos hiperparametros.

Implementa funciones que entrenan multiples SOMs variando un solo hiperparametro
(grid_size, init, radius_0, eta_0, seed) y producen figuras comparativas y
tablas resumen.

Las funciones devuelven los resultados (lista de dicts) y guardan figuras.
No imprimen tablas: el `main_kohonen_sweep.py` lo hace por separado.
"""

import numpy as np
import matplotlib.pyplot as plt

from shared.plotting import save_fig, PALETTE
from ej1.models.kohonen import SOM


def train_som(X, **kwargs):
    """Crea y entrena un SOM con los kwargs dados. Devuelve el SOM."""
    som = SOM(**kwargs)
    som.fit(X)
    return som


def som_summary(som, X, countries):
    """Devuelve dict con metricas resumen de un SOM entrenado.

    qe_final: error de cuantizacion final.
    n_active: neuronas con al menos 1 pais.
    n_empty: neuronas sin paises.
    max_hits: maximo de paises en una misma neurona.
    mean_hits_active: paises promedio por neurona activa.
    u_mean: distancia promedio U (media de la matriz U).
    u_std: std de U (mayor std = clusters mas definidos / fronteras nitidas).
    """
    hits = som.hits(X)
    U = som.u_matrix()
    n_active = int((hits > 0).sum())
    n_empty = int((hits == 0).sum())
    max_hits = int(hits.max())
    mean_hits_active = float(hits.sum() / max(n_active, 1))
    return {
        "qe_final": float(som.history_[-1]["qe"]),
        "n_active": n_active,
        "n_empty": n_empty,
        "n_total": som.grid_size ** 2,
        "max_hits": max_hits,
        "mean_hits_active": mean_hits_active,
        "u_mean": float(U.mean()),
        "u_std": float(U.std()),
    }


def _draw_country_panel(ax, som, X, countries, title):
    """Dibuja en `ax` el mapa de paises (etiquetas) sobre fondo de hits."""
    k = som.grid_size
    hits = som.hits(X)
    assignments = {}
    for x, name in zip(X, countries):
        i, j = som._winner(x, som.W)
        assignments.setdefault((i, j), []).append(name)

    im = ax.imshow(hits, cmap="Blues", vmin=0, vmax=max(1, hits.max()))
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(j) for j in range(k)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(k)], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    for (i, j), names in assignments.items():
        text = "\n".join(names)
        color = "black" if hits[i, j] < hits.max() * 0.6 else "white"
        ax.text(j, i, text, ha="center", va="center",
                fontsize=6.2, color=color)
    return im


def _draw_u_panel(ax, som, title):
    """Dibuja en `ax` la matriz U como heatmap."""
    U = som.u_matrix()
    k = som.grid_size
    im = ax.imshow(U, cmap="bone_r")
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([str(j) for j in range(k)], fontsize=8)
    ax.set_yticklabels([str(i) for i in range(k)], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(k):
        for j in range(k):
            color = "black" if U[i, j] < U.max() * 0.6 else "white"
            ax.text(j, i, f"{U[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    return im


def sweep_grid_size(X, countries, base_cfg, grid_sizes, path_maps, path_summary):
    """Entrena un SOM por cada grid_size. Genera dos figuras:
        path_maps: panel con mapa de paises por cada grid_size.
        path_summary: barras comparando qe_final, n_active y u_mean.
    Devuelve la lista de (grid_size, summary) por configuracion.
    """
    results = []
    soms = []
    for k in grid_sizes:
        cfg = dict(base_cfg)
        cfg["grid_size"] = k
        # Adaptar radius_0 a k/2 para que sea comparable.
        cfg["radius_0"] = max(2.0, k / 2)
        som = train_som(X, **cfg)
        soms.append((k, som))
        results.append((k, som_summary(som, X, countries)))

    # Figura 1: paneles con los mapas
    n = len(grid_sizes)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, (k, som) in zip(axes, soms):
        _draw_country_panel(ax, som, X, countries, f"grid {k}x{k}  ({k*k} neuronas)")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Barrido de tamano de grilla", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, path_maps)

    # Figura 2: barras resumen
    _plot_sweep_summary(results, "grilla", path_summary,
                        param_format=lambda v: f"{v}x{v}")

    return results


def sweep_init(X, countries, base_cfg, inits, path_maps, path_summary, n_seeds=4):
    """Entrena SOMs variando init y promediando sobre seeds. Por cada init,
    entrena con `n_seeds` semillas distintas (empezando en base_cfg['seed'])
    y guarda summaries individuales + una figura con el primer seed.
    """
    results = []
    soms_for_plot = []
    base_seed = base_cfg.get("seed", 42)
    for init in inits:
        seed_summaries = []
        first_som = None
        for s in range(n_seeds):
            cfg = dict(base_cfg)
            cfg["init"] = init
            cfg["seed"] = base_seed + s * 17
            som = train_som(X, **cfg)
            seed_summaries.append(som_summary(som, X, countries))
            if s == 0:
                first_som = som
        # Promediar metricas sobre seeds.
        avg = {}
        for key in seed_summaries[0]:
            vals = [d[key] for d in seed_summaries]
            avg[key] = float(np.mean(vals))
            avg[key + "_std"] = float(np.std(vals))
        results.append((init, avg))
        soms_for_plot.append((init, first_som))

    # figsize ancho-bajo para que en el slide entre junto con la barra de resumen.
    fig, axes = plt.subplots(1, len(inits), figsize=(7.5 * len(inits), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (init, som) in zip(axes, soms_for_plot):
        _draw_country_panel(ax, som, X, countries, f"init = '{init}'  (seed={base_seed})")
    fig.suptitle(f"Barrido de inicializacion (promedio sobre {n_seeds} seeds en la tabla resumen)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, path_maps)

    _plot_sweep_summary(results, "init", path_summary, param_format=str,
                        show_std=True)

    return results


def sweep_radius(X, countries, base_cfg, radius_0_values, path_maps, path_summary):
    """Entrena SOMs variando radius_0 con radius_final fijo en 1.0.
    Tambien incluye la variante 'radio constante' (radius_final = radius_0).
    """
    results = []
    soms = []
    for r0 in radius_0_values:
        cfg = dict(base_cfg)
        cfg["radius_0"] = r0
        # Convencion: si r0 <= 1.0, dejamos radius_final = r0 para no subirlo.
        cfg["radius_final"] = min(1.0, r0)
        som = train_som(X, **cfg)
        soms.append((f"R0={r0:.1f}", som))
        results.append((f"R0={r0:.1f}", som_summary(som, X, countries)))

    # Configuracion extra: radius constante igual a 2.0 (no decae).
    cfg = dict(base_cfg)
    cfg["radius_0"] = 2.0
    cfg["radius_final"] = 2.0
    som = train_som(X, **cfg)
    soms.append(("R cte = 2", som))
    results.append(("R cte = 2", som_summary(som, X, countries)))

    n = len(soms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, (label, som) in zip(axes, soms):
        _draw_country_panel(ax, som, X, countries, label)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Barrido de radio inicial (radius_final = 1.0 salvo en R cte)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, path_maps)

    _plot_sweep_summary(results, "radius", path_summary, param_format=str)

    return results


def sweep_eta(X, countries, base_cfg, eta_0_values, path_maps, path_summary):
    """Entrena SOMs variando eta_0 con eta_final fijo en 0.01."""
    results = []
    soms = []
    for e0 in eta_0_values:
        cfg = dict(base_cfg)
        cfg["eta_0"] = e0
        som = train_som(X, **cfg)
        soms.append((f"eta_0={e0:.2f}", som))
        results.append((f"eta_0={e0:.2f}", som_summary(som, X, countries)))

    n = len(soms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, (label, som) in zip(axes, soms):
        _draw_country_panel(ax, som, X, countries, label)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Barrido de tasa de aprendizaje inicial (eta_final = 0.01)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, path_maps)

    _plot_sweep_summary(results, "eta_0", path_summary, param_format=str)

    return results


def sweep_seeds(X, countries, base_cfg, seeds, path_maps, path_summary):
    """Entrena el mismo SOM con seeds distintos para mostrar estabilidad / variabilidad."""
    results = []
    soms = []
    for s in seeds:
        cfg = dict(base_cfg)
        cfg["seed"] = s
        som = train_som(X, **cfg)
        soms.append((f"seed={s}", som))
        results.append((f"seed={s}", som_summary(som, X, countries)))

    n = len(soms)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.2 * rows))
    axes = np.atleast_2d(axes).flatten()
    for ax, (label, som) in zip(axes, soms):
        _draw_country_panel(ax, som, X, countries, label)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Variabilidad por seed (misma configuracion)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, path_maps)

    _plot_sweep_summary(results, "seed", path_summary, param_format=str)

    return results


def _plot_sweep_summary(results, param_name, path, param_format=str, show_std=False):
    """Tres paneles de barras: qe_final, neuronas activas, U mean.
    `results` es [(param_value, summary_dict), ...]."""
    labels = [param_format(p) for p, _ in results]
    qe = [s["qe_final"] for _, s in results]
    n_active = [s["n_active"] for _, s in results]
    u_mean = [s["u_mean"] for _, s in results]
    u_std = [s["u_std"] for _, s in results]

    qe_err = [s.get("qe_final_std", 0) for _, s in results] if show_std else None
    nact_err = [s.get("n_active_std", 0) for _, s in results] if show_std else None
    umean_err = [s.get("u_mean_std", 0) for _, s in results] if show_std else None

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    xs = np.arange(len(labels))

    axes[0].bar(xs, qe, yerr=qe_err, color=PALETTE["primary"],
                edgecolor="black", linewidth=0.4, capsize=4)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Quantization Error final")
    axes[0].set_title("QE final (menor = mejor ajuste)")
    axes[0].grid(axis="y", alpha=0.3)
    for x, v in zip(xs, qe):
        axes[0].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(xs, n_active, yerr=nact_err, color=PALETTE["accent"],
                edgecolor="black", linewidth=0.4, capsize=4)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Neuronas activas")
    axes[1].set_title("Neuronas activas (28 paises distribuidos)")
    axes[1].grid(axis="y", alpha=0.3)
    for x, v in zip(xs, n_active):
        axes[1].text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(xs, u_mean, yerr=umean_err, color=PALETTE["highlight"],
                edgecolor="black", linewidth=0.4, capsize=4)
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, rotation=15, ha="right")
    axes[2].set_ylabel("Distancia U promedio")
    axes[2].set_title("U mean (menor = mapa mas continuo)")
    axes[2].grid(axis="y", alpha=0.3)
    for x, v in zip(xs, u_mean):
        axes[2].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[3].bar(xs, u_std, color=PALETTE["positive"],
                edgecolor="black", linewidth=0.4)
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(labels, rotation=15, ha="right")
    axes[3].set_ylabel("std de la matriz U")
    axes[3].set_title("U std (mayor = fronteras mas marcadas)")
    axes[3].grid(axis="y", alpha=0.3)
    for x, v in zip(xs, u_std):
        axes[3].text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Resumen del barrido sobre '{param_name}'",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)
