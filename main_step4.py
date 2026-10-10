#!/usr/bin/env python3
"""
==============================================================================
main_step4.py — Étape 4 : Validation numérique du PINN
==============================================================================

Le sujet demande : « Développer un solveur classique (Différences Finies ou
Éléments Finis) pour obtenir une solution de référence et évaluer la
précision du PINN (MSE, L2-relative error). »

Ce script :
    1. VALIDE le solveur lui-même (un solveur de référence non vérifié ne
       prouve rien) : mode propre exact + vérification de l'ordre 2 en espace ;
    2. recharge le PINN entraîné à l'Étape 3 (checkpoint) ;
    3. calcule la solution de référence sur l'objet chaud du checkpoint ;
    4. compare PINN et référence : MSE, erreur L2 relative, globales et en
       fonction du temps, part de températures non physiques (T* < 0) ;
    5. exporte la figure `step4_validation.png` et `step4_metrics.json`.

Usage
-----
    python main_step3.py                         # (une fois) produit checkpoints/best_model.pt
    python main_step4.py
    python main_step4.py --checkpoint checkpoints/last_model.pt --n-grid 201
    python main_step4.py --solver-only           # valide uniquement le solveur DF
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import DataConfig
from src.fd_solver import solve_heat_fd
from src.inference import load_trained_model
from src.validation import plot_validation, validate_pinn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Étape 4 — validation numérique du PINN")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Checkpoint du PINN (défaut : checkpoints/best_model.pt)")
    parser.add_argument("--n-grid", type=int, default=101,
                        help="Nœuds par direction du solveur DF (défaut : 101)")
    parser.add_argument("--n-times", type=int, default=51,
                        help="Nombre d'instants comparés (défaut : 51)")
    parser.add_argument("--activation", default="tanh",
                        help="Activation à supposer si le checkpoint est ancien et ne la précise pas")
    parser.add_argument("--out", default="step4_validation.png", help="Figure PNG de sortie")
    parser.add_argument("--json", default="step4_metrics.json", help="Métriques JSON de sortie")
    parser.add_argument("--solver-only", action="store_true",
                        help="Ne valide que le solveur DF (pas de checkpoint requis)")
    return parser.parse_args()


def check_solver(cfg: DataConfig) -> bool:
    """
    Vérifie le solveur sur un cas à solution EXACTE.

    Le mode propre sin(πx)·sin(πy) se désintègre en exp(−2π² t*). L'erreur
    du schéma doit (a) être petite et (b) décroître en ~h² quand on raffine
    la grille : c'est la signature d'un schéma d'ordre 2 en espace.
    """
    print("[1/4] Validation du solveur aux différences finies (mode propre exact)")
    errors = []
    grid_sizes = (26, 51, 101)
    for n_grid in grid_sizes:
        nodes = np.linspace(0.0, 1.0, n_grid)
        grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
        mode = np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
        solution = solve_heat_fd(cfg, n_grid=n_grid, n_times=3, initial_condition=mode)
        exact = mode * np.exp(-2.0 * np.pi**2 * solution.t_star[-1])
        error = np.linalg.norm(solution.T_star[-1] - exact) / np.linalg.norm(exact)
        errors.append(error)
        print(f"  n={n_grid:4d}  h={1/(n_grid-1):.4f}  erreur L2 relative = {error:.3e}"
              f"  ({solution.n_steps} pas)")
    # Raffiner h par 2 doit diviser l'erreur par ~4 (ordre 2).
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    print(f"  ordre de convergence observé : {', '.join(f'{o:.2f}' for o in orders)} (attendu ≈ 2)")
    ok = errors[-1] < 1e-2 and all(order > 1.5 for order in orders)
    print(f"  [{'OK' if ok else 'ÉCHEC'}] solveur {'validé' if ok else 'NON validé'}")
    return ok


def main() -> int:
    args = parse_args()
    cfg = DataConfig()

    solver_ok = check_solver(cfg)
    if args.solver_only:
        return 0 if solver_ok else 1
    if not solver_ok:
        print("Le solveur de référence n'est pas fiable : arrêt.")
        return 1

    print(f"\n[2/4] Chargement du PINN : {args.checkpoint}")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        print(f"  [ERREUR] Checkpoint introuvable : {checkpoint.resolve()}\n"
              "  → Lancer d'abord `python main_step3.py` (ou `--test` pour un run court).")
        return 1
    model, trained_cfg, payload = load_trained_model(
        checkpoint, fallback_activation=args.activation
    )
    if trained_cfg is None:
        print("  [info] Checkpoint sans DataConfig : objet par défaut supposé "
              "(disque r=0.15 au centre).")
        trained_cfg = cfg
    print(model.summary())
    print(f"  Objet d'entraînement : {trained_cfg.obj_shape} @ "
          f"({trained_cfg.obj_cx}, {trained_cfg.obj_cy}), r={trained_cfg.obj_radius}")
    print(f"  Checkpoint : itération {payload.get('epoch')}, phase {payload.get('phase')}, "
          f"record {payload.get('best_loss')}")

    print(f"\n[3/4] Solution de référence (DF {args.n_grid}×{args.n_grid}, "
          f"{args.n_times} instants)…")
    reference = solve_heat_fd(trained_cfg, n_grid=args.n_grid, n_times=args.n_times)
    print(f"  [OK] {reference.n_steps} pas de temps explicites")
    report = validate_pinn(model, reference)

    print("\n[4/4] Métriques PINN vs référence")
    print(report.summary(delta_T=trained_cfg.delta_T))
    print("\n  t*      MSE         L2 rel.")
    for k in np.linspace(0, len(report.t_star) - 1, 6).astype(int):
        print(f"  {report.t_star[k]:.3f}   {report.mse_per_time[k]:.3e}   {report.l2_per_time[k]:.3e}")

    figure_path = plot_validation(report, args.out)
    print(f"\n  [OK] Figure → {figure_path}")
    metrics = report.as_dict()
    metrics["checkpoint"] = str(checkpoint)
    metrics["n_grid"] = args.n_grid
    Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  [OK] Métriques → {Path(args.json).resolve()}")

    print("\n✓ Étape 4 terminée.")
    print("  → Prochaine étape : démonstrateur interactif (python main_step5.py).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
