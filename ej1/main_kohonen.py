"""Entrypoint del bloque Kohonen (ejercicio 1.1).

Lee la configuracion, carga europe.csv, estandariza, entrena el SOM y genera
5 figuras en ej1/results/kohonen/.

Uso:
    python3 ej1/main_kohonen.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from shared.config_loader import load_yaml
from shared.io_utils import load_europe
from shared.preprocessing import ZScoreScaler

from ej1.models.kohonen import SOM
from ej1.analysis.kohonen import (
    assign_countries,
    plot_country_map,
    plot_u_matrix,
    plot_hits,
    plot_variable_maps,
    plot_convergence,
)


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    data_path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    out_dir = os.path.join(REPO_ROOT, cfg["results"]["kohonen"])
    koh_cfg = cfg["kohonen"]

    countries, X, feature_names = load_europe(data_path)
    scaler = ZScoreScaler().fit(X)
    X_std = scaler.transform(X)

    som = SOM(
        grid_size=koh_cfg["grid_size"],
        epochs=koh_cfg["epochs"],
        eta_0=koh_cfg["eta_0"],
        eta_final=koh_cfg["eta_final"],
        radius_0=koh_cfg["radius_0"],
        radius_final=koh_cfg["radius_final"],
        init=koh_cfg["init"],
        seed=koh_cfg["seed"],
    )
    som.fit(X_std)

    print("=" * 70)
    print("Kohonen / SOM - Resultados")
    print("=" * 70)
    print(f"Grilla: {koh_cfg['grid_size']}x{koh_cfg['grid_size']}  "
          f"epochs={koh_cfg['epochs']}  "
          f"eta=[{koh_cfg['eta_0']} -> {koh_cfg['eta_final']}]  "
          f"radius=[{koh_cfg['radius_0']} -> {koh_cfg['radius_final']}]")
    print()
    qe_inicial = som.history_[0]["qe"]
    qe_final = som.history_[-1]["qe"]
    print(f"Quantization Error: inicial={qe_inicial:.4f}  final={qe_final:.4f}  "
          f"(reduccion {(1 - qe_final / qe_inicial) * 100:.1f}%)")
    print()

    hits = som.hits(X_std)
    assignments = assign_countries(som, X_std, countries)
    total = int(hits.sum())
    n_active = int((hits > 0).sum())
    n_total_neurons = koh_cfg["grid_size"] ** 2
    print(f"Paises asignados: {total} de {len(countries)}")
    print(f"Neuronas activas: {n_active} / {n_total_neurons}")
    print()
    print("Asignacion pais -> neurona:")
    for (i, j), names in sorted(assignments.items()):
        print(f"  ({i}, {j}): {', '.join(names)}")
    print()

    # Sanity checks.
    # Nota: con init="samples", QE_inicial es artificialmente bajo (cada neurona
    # arranca igual a una muestra del dataset, asi que muchas muestras estan a
    # distancia ~0 de su ganadora). El check util es la estabilidad de QE en las
    # ultimas epocas, no comparar inicial vs final.
    assert total == len(countries), \
        f"Se perdieron paises en la asignacion: {total} vs {len(countries)}"
    assert not np.isnan(som.W).any(), "Hay NaN en los pesos finales"
    recent_qes = np.array([h["qe"] for h in som.history_[-20:]])
    qe_rel_std = float(recent_qes.std() / (recent_qes.mean() + 1e-12))
    assert qe_rel_std < 0.05, \
        f"QE no se estabilizo en las ultimas epocas (std rel = {qe_rel_std:.4f})"
    print(f"QE estabilizado (std relativo en ultimas 20 epocas = {qe_rel_std:.5f})")
    print()

    os.makedirs(out_dir, exist_ok=True)

    plot_country_map(som, X_std, countries,
                     os.path.join(out_dir, "kohonen_country_map.png"))
    plot_u_matrix(som, os.path.join(out_dir, "kohonen_u_matrix.png"))
    plot_hits(som, X_std, os.path.join(out_dir, "kohonen_hits_map.png"))
    plot_variable_maps(som, feature_names,
                       os.path.join(out_dir, "kohonen_variable_maps.png"))
    plot_convergence(som.history_,
                     os.path.join(out_dir, "kohonen_convergence.png"))

    print(f"Figuras guardadas en: {out_dir}")
    print("OK")


if __name__ == "__main__":
    main()
