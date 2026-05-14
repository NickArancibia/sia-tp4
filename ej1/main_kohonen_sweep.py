"""Barrido de hiperparametros del SOM (ejercicio 1.1, analisis ampliado).

Entrena multiples SOMs variando un solo hiperparametro a la vez y guarda:
- Una figura tipo grilla con los mapas de paises para cada configuracion.
- Una figura resumen con barras de QE final, neuronas activas, U mean / std.
- Una tabla impresa por stdout con los numeros.

Las salidas se guardan en ej1/results/kohonen/sweep/.

Uso:
    python3 ej1/main_kohonen_sweep.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from shared.config_loader import load_yaml
from shared.io_utils import load_europe
from shared.preprocessing import ZScoreScaler

from ej1.analysis.kohonen_sweep import (
    sweep_grid_size,
    sweep_init,
    sweep_radius,
    sweep_eta,
    sweep_seeds,
)


def _print_table(title, results, key_order):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    header = f"{'param':<14s}" + "".join(f"{k:>16s}" for k in key_order)
    print(header)
    print("-" * len(header))
    for param, summary in results:
        line = f"{str(param):<14s}"
        for k in key_order:
            v = summary.get(k, "")
            line += f"{v:>16.4f}" if isinstance(v, float) else f"{str(v):>16s}"
        print(line)


def main():
    cfg = load_yaml(os.path.join(REPO_ROOT, "ej1", "config.yaml"))
    data_path = os.path.join(REPO_ROOT, cfg["data"]["path"])
    out_dir = os.path.join(REPO_ROOT, cfg["results"]["kohonen"], "sweep")
    os.makedirs(out_dir, exist_ok=True)

    countries, X, feature_names = load_europe(data_path)
    scaler = ZScoreScaler().fit(X)
    X_std = scaler.transform(X)

    base_cfg = dict(cfg["kohonen"])

    # Barrido 1: grid_size.
    print("Corriendo barrido: grid_size...")
    r1 = sweep_grid_size(
        X_std, countries, base_cfg,
        grid_sizes=[3, 4, 5, 6],
        path_maps=os.path.join(out_dir, "sweep_grid_size_maps.png"),
        path_summary=os.path.join(out_dir, "sweep_grid_size_summary.png"),
    )
    _print_table("Barrido grid_size", r1,
                 ["qe_final", "n_active", "n_empty", "u_mean", "u_std"])

    # Barrido 2: init.
    print("Corriendo barrido: init (samples vs random, promediado sobre seeds)...")
    r2 = sweep_init(
        X_std, countries, base_cfg,
        inits=["samples", "random"],
        path_maps=os.path.join(out_dir, "sweep_init_maps.png"),
        path_summary=os.path.join(out_dir, "sweep_init_summary.png"),
        n_seeds=4,
    )
    _print_table("Barrido init (promedio sobre seeds)", r2,
                 ["qe_final", "n_active", "n_empty", "u_mean", "u_std"])

    # Barrido 3: radius_0 + variante de radio constante.
    print("Corriendo barrido: radius_0...")
    r3 = sweep_radius(
        X_std, countries, base_cfg,
        radius_0_values=[1.0, 2.0, 3.0],
        path_maps=os.path.join(out_dir, "sweep_radius_maps.png"),
        path_summary=os.path.join(out_dir, "sweep_radius_summary.png"),
    )
    _print_table("Barrido radius_0 (radius_final = 1 salvo en R cte)", r3,
                 ["qe_final", "n_active", "n_empty", "u_mean", "u_std"])

    # Barrido 4: eta_0.
    print("Corriendo barrido: eta_0...")
    r4 = sweep_eta(
        X_std, countries, base_cfg,
        eta_0_values=[0.1, 0.3, 0.5, 0.9],
        path_maps=os.path.join(out_dir, "sweep_eta_maps.png"),
        path_summary=os.path.join(out_dir, "sweep_eta_summary.png"),
    )
    _print_table("Barrido eta_0", r4,
                 ["qe_final", "n_active", "n_empty", "u_mean", "u_std"])

    # Barrido 5: seeds (estabilidad de la configuracion base).
    print("Corriendo barrido: seeds...")
    r5 = sweep_seeds(
        X_std, countries, base_cfg,
        seeds=[42, 7, 123, 1000],
        path_maps=os.path.join(out_dir, "sweep_seeds_maps.png"),
        path_summary=os.path.join(out_dir, "sweep_seeds_summary.png"),
    )
    _print_table("Barrido seeds (config base)", r5,
                 ["qe_final", "n_active", "n_empty", "u_mean", "u_std"])

    print()
    print(f"Figuras del barrido guardadas en: {out_dir}")
    print("OK")


if __name__ == "__main__":
    main()
