"""Entrypoint del bloque PCA con libreria (ejercicio 1, sub-entrega prioritaria).

Lee la configuracion, carga europe.csv, estandariza, corre PCA y genera 8 figuras
en ej1/results/pca/.

Uso:
    python3 ej1/main_pca.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from shared.config_loader import load_yaml
from shared.io_utils import load_europe
from shared.preprocessing import ZScoreScaler

from ej1.models.pca import run_pca
from ej1.analysis.pca import (
    plot_boxplots,
    plot_correlation_heatmap,
    plot_scree,
    plot_loadings_heatmap,
    plot_pc1_pc2_loadings_bars,
    plot_biplot,
    plot_pc_ranking,
    summarize_pc1_interpretation,
)


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    data_path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    out_dir = os.path.join(REPO_ROOT, cfg["results"]["pca"])
    threshold = float(cfg["pca"].get("variance_threshold", 0.8))

    countries, X, feature_names = load_europe(data_path)
    scaler = ZScoreScaler().fit(X)
    X_std = scaler.transform(X)

    res = run_pca(X_std)

    # Sanity checks
    assert abs(res["explained_variance_ratio"].sum() - 1.0) < 1e-9, \
        "La suma de explained_variance_ratio no es 1"

    print("=" * 70)
    print("PCA - Resultados")
    print("=" * 70)
    print(f"Dataset: {len(countries)} paises x {len(feature_names)} variables")
    print(f"Features: {feature_names}")
    print()
    print("Varianza explicada por componente:")
    for i, (lam, evr, cum) in enumerate(zip(
            res["eigenvalues"],
            res["explained_variance_ratio"],
            res["explained_variance_cum"])):
        print(f"  PC{i + 1}: eigval={lam:.4f}  evr={evr:.4f}  cum={cum:.4f}")
    print()

    for pc_idx in [0, 1]:
        interp = summarize_pc1_interpretation(res["loadings"], feature_names, pc_index=pc_idx)
        print(f"Interpretacion de PC{pc_idx+1} (cargas ordenadas por magnitud):")
        for name, val in interp["loadings_sorted_by_abs"]:
            print(f"  {name:>14s}: {val:+.4f}")
        print()
        top = np.argsort(res["scores"][:, pc_idx])[::-1][:5]
        bot = np.argsort(res["scores"][:, pc_idx])[:5]
        print(f"Top 5 paises por PC{pc_idx+1}: {[countries[i] for i in top]}")
        print(f"Bot 5 paises por PC{pc_idx+1}: {[countries[i] for i in bot]}")
        print()

    os.makedirs(out_dir, exist_ok=True)

    # Guardar cargas (loadings) en CSV
    loadings_df = pd.DataFrame(
        res["loadings"],
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(res["loadings"].shape[1])]
    )
    loadings_path = os.path.join(out_dir, "pca_loadings.csv")
    loadings_df.to_csv(loadings_path)
    print(f"Cargas guardadas en: {loadings_path}")

    # Guardar scores (proyecciones de cada pais) en CSV
    scores_df = pd.DataFrame(
        res["scores"],
        index=countries,
        columns=[f"PC{i+1}" for i in range(res["scores"].shape[1])]
    )
    scores_path = os.path.join(out_dir, "pca_scores.csv")
    scores_df.to_csv(scores_path)
    print(f"Scores guardados en: {scores_path}")
    print()

    plot_boxplots(X, feature_names,
                  os.path.join(out_dir, "boxplot_raw.png"),
                  title="Variables originales sin estandarizar",
                  standardized=False)
    plot_boxplots(X_std, feature_names,
                  os.path.join(out_dir, "boxplot_standardized.png"),
                  title="Variables estandarizadas (z-score)",
                  standardized=True)
    plot_correlation_heatmap(X_std, feature_names,
                             os.path.join(out_dir, "correlation_heatmap.png"))
    plot_scree(res["explained_variance_ratio"],
               os.path.join(out_dir, "scree_plot.png"),
               threshold=threshold)
    plot_loadings_heatmap(res["loadings"], feature_names,
                          os.path.join(out_dir, "loadings_heatmap.png"))
    plot_pc1_pc2_loadings_bars(res["loadings"], feature_names,
                               os.path.join(out_dir, "pc1_pc2_loadings_bars.png"))
    plot_biplot(res["scores"], res["loadings"], countries, feature_names,
                os.path.join(out_dir, "biplot_pc1_pc2.png"),
                pc_x=0, pc_y=1)
    plot_pc_ranking(res["scores"][:, 0], countries,
                    os.path.join(out_dir, "pc1_ranking.png"),
                    pc_index=0)
    plot_pc_ranking(res["scores"][:, 1], countries,
                    os.path.join(out_dir, "pc2_ranking.png"),
                    pc_index=1)

    print(f"Figuras guardadas en: {out_dir}")
    print("OK")


if __name__ == "__main__":
    main()
