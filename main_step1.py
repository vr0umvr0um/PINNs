#!/usr/bin/env python3
"""
Étape 1 — Adimensionnement & Échantillonnage
============================================

Génère les points de collocation (IC, BC, résidu) pour le PINN de
diffusion thermique 2D instationnaire, vérifie les invariants
mathématiques, et exporte une figure de diagnostic.

Usage
-----
    python main_step1.py
    python main_step1.py --method lhs --seed 123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DataConfig
from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import set_seed

# ---------------------------------------------------------------------------
# Assertions / validation
# ---------------------------------------------------------------------------

def validate_ic(ic: dict, cfg: DataConfig) -> None:
    """Vérifie formes, t*=0 et requires_grad sur les coordonnées IC."""
    n = cfg.N_ic
    for key in ("x_star", "y_star", "t_star", "T_star"):
        assert key in ic, f"IC missing key {key}"
        assert ic[key].shape == (n, 1), f"IC[{key}] shape {ic[key].shape} != ({n}, 1)"

    assert torch.allclose(
        ic["t_star"], torch.zeros_like(ic["t_star"])
    ), "IC: t* must be identically 0"
    assert ic["x_star"].requires_grad, "IC: x_star must require grad"
    assert ic["y_star"].requires_grad, "IC: y_star must require grad"
    assert ic["t_star"].requires_grad, "IC: t_star must require grad"

    x = ic["x_star"].detach()
    y = ic["y_star"].detach()
    assert (x >= 0).all() and (x <= 1).all(), "IC: x* out of [0, 1]"
    assert (y >= 0).all() and (y <= 1).all(), "IC: y* out of [0, 1]"
    print(f"  [OK] IC  : N={n}, t*=0, T*∈[{ic['T_star'].min():.2f}, {ic['T_star'].max():.2f}]")


def validate_bc(bc: dict, cfg: DataConfig) -> None:
    """Vérifie formes, T*=0 et position sur les parois."""
    n = cfg.N_bc
    for key in ("x_star", "y_star", "t_star", "T_star"):
        assert key in bc, f"BC missing key {key}"
        assert bc[key].shape == (n, 1), f"BC[{key}] shape {bc[key].shape} != ({n}, 1)"

    assert torch.allclose(
        bc["T_star"], torch.zeros_like(bc["T_star"])
    ), "BC: T* must be identically 0 (Dirichlet ambiante)"
    assert bc["x_star"].requires_grad, "BC: x_star must require grad"
    assert bc["y_star"].requires_grad, "BC: y_star must require grad"
    assert bc["t_star"].requires_grad, "BC: t_star must require grad"

    x = bc["x_star"].detach().cpu().numpy().ravel()
    y = bc["y_star"].detach().cpu().numpy().ravel()
    on_left = np.isclose(x, 0.0)
    on_right = np.isclose(x, 1.0)
    on_bottom = np.isclose(y, 0.0)
    on_top = np.isclose(y, 1.0)
    on_boundary = on_left | on_right | on_bottom | on_top
    assert on_boundary.all(), "BC: every point must lie on the domain boundary"

    print(
        f"  [OK] BC  : N={n}, T*=0, "
        f"walls L/R/B/T = {on_left.sum()}/{on_right.sum()}/{on_bottom.sum()}/{on_top.sum()}"
    )


def validate_residual(res: dict, cfg: DataConfig) -> None:
    """Vérifie formes et appartenance au cube ouvert ]0, 1[³."""
    n = cfg.N_res
    for key in ("x_star", "y_star", "t_star"):
        assert key in res, f"RES missing key {key}"
        assert res[key].shape == (n, 1), f"RES[{key}] shape {res[key].shape} != ({n}, 1)"
        assert res[key].requires_grad, f"RES: {key} must require grad"
        v = res[key].detach()
        assert (v > 0).all() and (v < 1).all(), f"RES: {key} must lie in (0, 1)"

    print(f"  [OK] RES : N={n}, points strictly interior to (0,1)³")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_collocation_points(
    ic: dict,
    bc: dict,
    res: dict,
    cfg: DataConfig,
    out_path: Path,
) -> None:
    """
    Figure diagnostic 2×2 :
        (0,0) nuage (x*, y*) — IC / BC / résidu (sous-échantillon)
        (0,1) histogramme t* (BC + résidu)
        (1,0) répartition par paroi (BC)
        (1,1) résumé texte adimensionnement
    """
    # Sous-échantillon résidu pour lisibilité
    n_plot = min(4000, cfg.N_res)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(cfg.N_res, size=n_plot, replace=False)

    ic_x = ic["x_star"].detach().cpu().numpy().ravel()
    ic_y = ic["y_star"].detach().cpu().numpy().ravel()
    bc_x = bc["x_star"].detach().cpu().numpy().ravel()
    bc_y = bc["y_star"].detach().cpu().numpy().ravel()
    bc_t = bc["t_star"].detach().cpu().numpy().ravel()
    res_x = res["x_star"].detach().cpu().numpy().ravel()[idx]
    res_y = res["y_star"].detach().cpu().numpy().ravel()[idx]
    res_t = res["t_star"].detach().cpu().numpy().ravel()
    bc_wall = bc["wall"].detach().cpu().numpy().ravel()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Étape 1 — Adimensionnement & Échantillonnage\n"
        r"$\partial T^*/\partial t^* = \partial^2 T^*/\partial x^{*2} + \partial^2 T^*/\partial y^{*2}$"
        f"   |   domaine $(x^*, y^*)\\in[0,1]^2$",
        fontsize=13,
        fontweight="bold",
    )

    # --- (0,0) Spatial scatter ---
    ax = axes[0, 0]
    ax.scatter(res_x, res_y, s=2, c="#4C72B0", alpha=0.25, label=f"Résidu (n={n_plot}/{cfg.N_res})")
    ax.scatter(ic_x, ic_y, s=8, c="#55A868", alpha=0.7, label=f"IC t*=0 (n={cfg.N_ic})")
    ax.scatter(bc_x, bc_y, s=10, c="#C44E52", alpha=0.8, label=f"BC T*=0 (n={cfg.N_bc})")
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$y^*$")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title("Points de collocation (projection spatiale)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # --- (0,1) Histogramme t* ---
    ax = axes[0, 1]
    ax.hist(res_t, bins=40, color="#4C72B0", alpha=0.7, label="Résidu", density=True)
    ax.hist(bc_t, bins=40, color="#C44E52", alpha=0.6, label="BC", density=True)
    ax.axvline(0.0, color="#55A868", linewidth=2, linestyle="--", label="IC (t*=0)")
    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel("densité")
    ax.set_title("Distribution temporelle")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,0) Répartition parois BC ---
    ax = axes[1, 0]
    wall_names = ["Gauche\n(x*=0)", "Droite\n(x*=1)", "Bas\n(y*=0)", "Haut\n(y*=1)"]
    counts = [int((bc_wall == w).sum()) for w in range(4)]
    colors = ["#E07A5F", "#3D405B", "#81B29A", "#F2CC8F"]
    bars = ax.bar(wall_names, counts, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("nombre de points")
    ax.set_title("Répartition BC par paroi")
    ax.set_ylim(0, max(counts) * 1.2 if counts else 1)
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            str(c),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)

    # --- (1,1) Résumé adimensionnement ---
    ax = axes[1, 1]
    ax.axis("off")
    summary = (
        "Adimensionnement\n"
        "─────────────────────────────\n"
        f"  x* = x / Lx          Lx = {cfg.Lx} m\n"
        f"  y* = y / Ly          Ly = {cfg.Ly} m\n"
        f"  t* = t / t_max       t_max = {cfg.t_max} s\n"
        f"  T* = (T − T_amb)/ΔT  ΔT = {cfg.delta_T}°C\n"
        "\n"
        "Physique\n"
        "─────────────────────────────\n"
        f"  α  = {cfg.alpha:.2e} m²/s\n"
        f"  T_amb = {cfg.T_amb}°C   T_obj = {cfg.T_obj}°C\n"
        f"  t_ref = L²/α = {cfg.t_ref:.1f} s\n"
        f"  Fo_max = α t_max / L² = {cfg.t_star_max:.4f}\n"
        "\n"
        "Échantillonnage\n"
        "─────────────────────────────\n"
        f"  N_ic  = {cfg.N_ic:>6d}   (t* = 0, T* = 0)\n"
        f"  N_bc  = {cfg.N_bc:>6d}   (parois, T* = 0)\n"
        f"  N_res = {cfg.N_res:>6d}   (intérieur)\n"
        f"  seed  = {cfg.seed}\n"
        "\n"
        "Équation cible (adim.)\n"
        "─────────────────────────────\n"
        "  ∂T*/∂t* = ∂²T*/∂x*² + ∂²T*/∂y*²\n"
    )
    ax.text(
        0.05,
        0.95,
        summary,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Figure sauvegardée → {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PINN Thermal 2D — Étape 1 : échantillonnage")
    p.add_argument("--method", choices=["sobol", "lhs", "uniform"], default="sobol")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-ic", type=int, default=None)
    p.add_argument("--n-bc", type=int, default=None)
    p.add_argument("--n-res", type=int, default=None)
    p.add_argument("--out", type=str, default="collocation_points.png")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cfg_kwargs = {
        "seed": args.seed,
        "device": args.device,
    }
    if args.n_ic is not None:
        cfg_kwargs["N_ic"] = args.n_ic
    if args.n_bc is not None:
        cfg_kwargs["N_bc"] = args.n_bc
    if args.n_res is not None:
        cfg_kwargs["N_res"] = args.n_res

    cfg = DataConfig(**cfg_kwargs)
    set_seed(cfg.seed)

    print(cfg.summary())
    print(f"\nMéthode d'échantillonnage : {args.method}")
    print("\n[1/4] Génération des points…")

    ic = sample_ic(cfg, method=args.method)
    bc = sample_bc(cfg, method=args.method)
    res = sample_residual(cfg, method=args.method)

    print("[2/4] Validation des invariants…")
    validate_ic(ic, cfg)
    validate_bc(bc, cfg)
    validate_residual(res, cfg)

    print("[3/4] Export de la figure diagnostic…")
    plot_collocation_points(ic, bc, res, cfg, out_path=Path(args.out))

    print("[4/4] Récapitulatif des tenseurs :")
    for name, bundle in ("IC", ic), ("BC", bc), ("RES", res):
        print(f"  {name}:")
        for k, v in bundle.items():
            rg = getattr(v, "requires_grad", False)
            print(f"    {k:8s} shape={tuple(v.shape)}  dtype={v.dtype}  requires_grad={rg}")

    print("\n✓ Étape 1 terminée avec succès.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
