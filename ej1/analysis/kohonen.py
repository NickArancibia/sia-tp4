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


def cluster_profile_table(som, X_raw, X_std, countries, feature_names):
    """Devuelve una lista de dicts, uno por neurona activa, con:
        - coordenadas (i, j) en la grilla.
        - paises asignados.
        - hits (cantidad de paises).
        - perfil promedio en valores ORIGINALES (no estandarizados) de las
          variables, computado sobre los paises caidos en la neurona.

    Permite explicar "por que estos paises son parecidos": muestra los valores
    crudos de Area, GDP, Inflation, etc. para cada cluster.
    """
    profiles = []
    k = som.grid_size
    for i in range(k):
        for j in range(k):
            members_idx = []
            for idx, x in enumerate(X_std):
                wi, wj = som._winner(x, som.W)
                if (wi, wj) == (i, j):
                    members_idx.append(idx)
            if not members_idx:
                continue
            members_idx = np.array(members_idx)
            profile_raw = X_raw[members_idx].mean(axis=0)
            profiles.append({
                "ij": (i, j),
                "countries": [countries[idx] for idx in members_idx],
                "hits": len(members_idx),
                "profile_raw": profile_raw,
                "features": list(feature_names),
            })
    return profiles


def plot_cluster_profiles(profiles, feature_names, path):
    """Tabla visual: una fila por cluster, columnas = variables.
    Cada celda se colorea con un heatmap divergente respecto a la media global
    para evidenciar como se diferencia cada cluster del promedio europeo.
    """
    if not profiles:
        return
    n = len(profiles)
    n_feat = len(feature_names)
    table = np.array([p["profile_raw"] for p in profiles])  # (n_clusters, n_feat)
    global_mean = table.mean(axis=0)
    global_std = table.std(axis=0) + 1e-12
    z = (table - global_mean) / global_std

    row_labels = []
    for p in profiles:
        i, j = p["ij"]
        names = ", ".join(p["countries"])
        if len(names) > 55:
            names = names[:52] + "..."
        row_labels.append(f"({i},{j}) [n={p['hits']}]  {names}")

    fig, ax = plt.subplots(figsize=(1.4 * n_feat + 4, 0.42 * n + 1.8))
    vmax = float(np.abs(z).max())
    im = ax.imshow(z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(feature_names, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    for i in range(n):
        for j in range(n_feat):
            v_raw = table[i, j]
            ax.text(j, i, f"{v_raw:.1f}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if abs(z[i, j]) > 0.7 * vmax else "black")
    ax.set_title("Perfil promedio por cluster (valores originales). "
                 "Color: z-score respecto al promedio entre clusters.",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-score entre clusters")
    fig.tight_layout()
    save_fig(fig, path)


def plot_neighbor_distance_graph(som, X, countries, path):
    """Grafico tipo nodo-arista: dibuja la grilla con neuronas activas como nodos
    (etiquetados con los paises) y aristas entre vecinos con grosor / color
    inversamente proporcional a la distancia U entre esos vecinos.
    Aristas mas gruesas = neuronas vecinas mas parecidas.
    """
    k = som.grid_size
    hits = som.hits(X)
    assignments = {}
    for x, name in zip(X, countries):
        i, j = som._winner(x, som.W)
        assignments.setdefault((i, j), []).append(name)

    fig, ax = plt.subplots(figsize=(max(9, 2.0 * k), max(8, 1.9 * k)))
    # Aristas: para cada par de vecinos 4-conexos, dibujar segmento.
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    # Calcular todas las distancias para normalizar grosor.
    dists_pairs = []
    for i in range(k):
        for j in range(k):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < k and 0 <= nj < k:
                    d = float(np.linalg.norm(som.W[i, j] - som.W[ni, nj]))
                    dists_pairs.append(((i, j), (ni, nj), d))
    if not dists_pairs:
        return
    d_min = min(d for _, _, d in dists_pairs)
    d_max = max(d for _, _, d in dists_pairs)
    for (i, j), (ni, nj), d in dists_pairs:
        # Normalizar: distancias chicas -> linea gruesa y oscura (mas parecidos).
        alpha = 1.0 - (d - d_min) / (d_max - d_min + 1e-12)
        lw = 0.5 + 4.5 * alpha
        color = plt.get_cmap("viridis")(alpha)
        ax.plot([j, nj], [i, ni], color=color, linewidth=lw, zorder=1, alpha=0.75)

    # Nodos: circulo cuyo tamano depende de los hits.
    for i in range(k):
        for j in range(k):
            h = hits[i, j]
            size = 380 if h == 0 else 480 + 220 * h
            facecolor = "lightgray" if h == 0 else "#d8e2dc"
            ax.scatter(j, i, s=size, facecolor=facecolor, edgecolor="black",
                       linewidth=0.6, zorder=2)
            if h > 0:
                names = assignments.get((i, j), [])
                txt = "\n".join(names)
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.8, zorder=3)

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels([f"col {j}" for j in range(k)])
    ax.set_yticklabels([f"fila {i}" for i in range(k)])
    ax.set_xlim(-0.7, k - 0.3)
    ax.set_ylim(k - 0.3, -0.7)  # invertir Y para que (0,0) este arriba
    ax.set_aspect("equal")
    ax.set_title("Distancias entre neuronas vecinas (aristas mas gruesas = vecinos mas parecidos)\n"
                 "El grosor codifica la distancia U; nodos vacios = neuronas sin paises",
                 fontsize=11)

    # Colorbar manual: tomamos un ScalarMappable.
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=d_min, vmax=d_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Distancia U (estandarizada)")
    # Invertir colorbar visualmente: la parte alta del cmap corresponde a d=d_min.
    cbar.ax.invert_yaxis()

    fig.tight_layout()
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
