"""Entrypoint del bloque Oja (ejercicio 1.2).

Lee la configuracion, carga europe.csv, estandariza, entrena Oja, calcula
PC1 con sklearn para comparar, alinea signos, computa metricas y genera
5 figuras en ej1/results/oja/.

Uso:
    python3 ej1/main_oja.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.config_loader import load_yaml
from shared.io_utils import load_europe
from shared.preprocessing import ZScoreScaler

from ej1.models.pca import run_pca
from ej1.models.oja import Oja
from ej1.analysis.oja import (
    align_sign,
    comparison_metrics,
    plot_convergence,
    plot_weights_evolution,
    plot_comparison_loadings,
    plot_comparison_scores,
    plot_oja_ranking,
)


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    data_path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    out_dir = os.path.join(REPO_ROOT, cfg["results"]["oja"])
    oja_cfg = cfg["oja"]

    countries, X, feature_names = load_europe(data_path)
    scaler = ZScoreScaler().fit(X)
    X_std = scaler.transform(X)

    # Referencia: PC1 obtenida con libreria, ya con signo normalizado.
    pca_res = run_pca(X_std)
    pc1_ref = pca_res["loadings"][:, 0]
    scores_ref = pca_res["scores"][:, 0]

    # Entrenamiento Oja, pasando la referencia para trackear cos_sim por epoca.
    model = Oja(
        eta=oja_cfg["eta"],
        epochs=oja_cfg["epochs"],
        seed=oja_cfg["seed"],
        init=oja_cfg["init"],
        init_unit_norm=oja_cfg["init_unit_norm"],
    )
    model.fit(X_std, reference=pc1_ref)

    # Alinear signo: Oja puede converger a -PC1 (es valido). Para comparar,
    # forzamos signo positivo respecto a la referencia.
    w_oja = align_sign(model.w_, pc1_ref)
    w_history_aligned = np.array([
        align_sign(w, pc1_ref) for w in model.w_history_
    ])
    scores_oja = X_std @ w_oja

    metrics = comparison_metrics(w_oja, pc1_ref, scores_oja, scores_ref)

    print("=" * 70)
    print("Oja - Resultados")
    print("=" * 70)
    print(f"Hiperparametros: eta={oja_cfg['eta']}  epochs={oja_cfg['epochs']}  "
          f"seed={oja_cfg['seed']}  init={oja_cfg['init']}")
    print()
    print("w_oja (alineado en signo con PC1 libreria):")
    for name, val in zip(feature_names, w_oja):
        print(f"  {name:>14s}: {val:+.6f}")
    print()
    print("Metricas de comparacion Oja vs libreria:")
    for k, v in metrics.items():
        print(f"  {k:>22s}: {v:+.8f}" if isinstance(v, float) else f"  {k:>22s}: {v}")
    print()

    # Sanity checks
    assert metrics["cos_sim"] > 0.999, \
        f"cos_sim({metrics['cos_sim']}) menor a 0.999"
    assert abs(metrics["norm_oja"] - 1.0) < 0.05, \
        f"||w_oja|| ({metrics['norm_oja']}) lejos de 1"
    assert metrics["max_abs_error_loadings"] < 0.01, \
        f"max_abs_error_loadings ({metrics['max_abs_error_loadings']}) > 0.01"

    os.makedirs(out_dir, exist_ok=True)

    plot_convergence(model.history_,
                     os.path.join(out_dir, "oja_convergence.png"))
    plot_weights_evolution(w_history_aligned, feature_names,
                           os.path.join(out_dir, "oja_weights_evolution.png"),
                           reference=pc1_ref)
    plot_comparison_loadings(w_oja, pc1_ref, feature_names,
                             os.path.join(out_dir, "oja_vs_library_loadings.png"))
    plot_comparison_scores(scores_oja, scores_ref, countries,
                           os.path.join(out_dir, "oja_vs_library_scores.png"))
    plot_oja_ranking(scores_oja, countries,
                     os.path.join(out_dir, "oja_pc1_ranking.png"))

    print(f"Figuras guardadas en: {out_dir}")
    print("OK")


if __name__ == "__main__":
    main()
