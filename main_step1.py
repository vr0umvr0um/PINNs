#!/usr/bin/env python3
"""
==============================================================================
main_step1.py — Étape 1 : Adimensionnement & Échantillonnage
==============================================================================

Rôle de ce script dans le projet
--------------------------------
C'est le POINT D'ENTRÉE de l'Étape 1. Il enchaîne quatre actions :

    1. Lire la configuration physique (DataConfig)
    2. Générer les 3 familles de points de collocation (IC, BC, résidu)
    3. Valider les invariants mathématiques (assertions pédagogiques)
    4. Exporter une figure diagnostic `collocation_points.png`

Pourquoi valider ici (et pas seulement dans les tests) ?
    Parce qu'en soutenance / démo live, on veut VOIR dans le terminal
    que t*=0, T*∈[0,1], t*∈[0, t*_max] AVANT même d'ouvrir la figure.
    Les assertions plantent le script si un bug d'échantillonnage revient.

Usage
-----
    python main_step1.py
    python main_step1.py --method lhs --seed 123
    python main_step1.py --n-ic 500 --n-bc 500 --n-res 2000   # run rapide
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from config import DataConfig
from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import set_seed

# Alias de type : un "batch" = dict nom → tenseur (sortie de sample_*)
CollocationBatch = Dict[str, torch.Tensor]


# ==========================================================================
# Helpers numériques pour les assertions
# ==========================================================================

def _is_close_to(value: float, target: float, tolerance: float = 1e-6) -> bool:
    """
    Test |value - target| ≤ tolerance.

    POURQUOI une tolérance et pas `==` ?
        Les float32/float64 ne sont jamais exacts après des opérations
        (scaling, conversions NumPy→Torch). Comparer strictement
        `T_max == 1.0` peut échouer pour 0.99999994.
    """
    return abs(value - target) <= tolerance


# ==========================================================================
# Validation des invariants (le "contrat" de l'Étape 1)
# ==========================================================================

def validate_ic(ic_batch: CollocationBatch, cfg: DataConfig) -> None:
    """
    Vérifie que le batch de condition initiale respecte le cahier des charges.

    Invariants contrôlés (à savoir expliquer en soutenance)
    -------------------------------------------------------
    1. Clés présentes : x_star, y_star, t_star, T_star
    2. Shape de chaque tenseur = (N_ic, 1)
       → format MLP : (batch_size, n_features=1)
    3. t* identiquement 0
       → définition même d'une condition "initiale"
    4. requires_grad=True sur x*, y*, t*
       → indispensable pour l'autograd du résidu à l'Étape 2
    5. (x*, y*) ∈ [0, 1]²
    6. T* ∈ [0, 1] avec les DEUX bornes atteintes
       → prouve que l'objet chaud (T*=1) ET l'extérieur (T*=0) sont peints

    Parameters
    ----------
    ic_batch : dict[str, Tensor]
        Sortie de sample_ic().
    cfg : DataConfig
        Fournit N_ic (taille attendue).
    """
    n_ic = cfg.N_ic
    expected_shape = (n_ic, 1)

    # --- Présence des clés et shapes -----------------------------------
    for key in ("x_star", "y_star", "t_star", "T_star"):
        assert key in ic_batch, f"IC : clé manquante '{key}'"
        actual_shape = tuple(ic_batch[key].shape)
        assert actual_shape == expected_shape, (
            f"IC['{key}'] shape {actual_shape} ≠ {expected_shape}"
        )

    # --- t* = 0 (invariant temporel de l'IC) ---------------------------
    assert torch.allclose(
        ic_batch["t_star"], torch.zeros_like(ic_batch["t_star"])
    ), "IC : t* doit être identiquement 0"

    # --- Autograd activé sur les coordonnées ---------------------------
    assert ic_batch["x_star"].requires_grad, "IC : x_star doit avoir requires_grad=True"
    assert ic_batch["y_star"].requires_grad, "IC : y_star doit avoir requires_grad=True"
    assert ic_batch["t_star"].requires_grad, "IC : t_star doit avoir requires_grad=True"

    # --- Domaine spatial -----------------------------------------------
    # .detach() coupe le graphe autograd : on lit les valeurs sans
    # construire de dérivées (plus rapide, et on n'en a pas besoin ici).
    x_star = ic_batch["x_star"].detach()
    y_star = ic_batch["y_star"].detach()
    T_star = ic_batch["T_star"].detach()

    assert (x_star >= 0).all() and (x_star <= 1).all(), "IC : x* hors de [0, 1]"
    assert (y_star >= 0).all() and (y_star <= 1).all(), "IC : y* hors de [0, 1]"

    # --- Champ T* : objet chaud (1) + extérieur (0) --------------------
    T_min = float(T_star.min())
    T_max = float(T_star.max())
    assert _is_close_to(T_min, 0.0), (
        f"IC : T* min attendu ≈ 0 (extérieur), reçu {T_min}"
    )
    assert _is_close_to(T_max, 1.0), (
        f"IC : T* max attendu ≈ 1 (objet chaud), reçu {T_max}. "
        "Vérifier la géométrie de l'objet / le masque dans sample_ic."
    )

    # Comptage pédagogique pour le log
    n_hot = int((T_star > 0.5).sum())  # points dans l'objet
    n_cold = n_ic - n_hot  # points hors objet
    print(
        f"  [OK] IC  : N={n_ic}, t*=0, T*∈[{T_min:.2f}, {T_max:.2f}]  "
        f"(hot={n_hot}, cold={n_cold})"
    )


def validate_bc(bc_batch: CollocationBatch, cfg: DataConfig) -> None:
    """
    Vérifie que le batch de conditions aux limites respecte le cahier des charges.

    Invariants contrôlés
    --------------------
    1. Clés + shapes (N_bc, 1)
    2. T* = 0 partout (Dirichlet ambiante)
    3. requires_grad=True sur x*, y*, t*
    4. Chaque point est sur UNE des 4 parois (x*=0/1 ou y*=0/1)
    5. t* ∈ [0, t*_max]   ← pas [0, 1] ! (bug Fourier classique)

    Parameters
    ----------
    bc_batch : dict[str, Tensor]
        Sortie de sample_bc().
    cfg : DataConfig
        Fournit N_bc et t_star_max.
    """
    n_bc = cfg.N_bc
    t_star_max = cfg.t_star_max
    expected_shape = (n_bc, 1)

    for key in ("x_star", "y_star", "t_star", "T_star"):
        assert key in bc_batch, f"BC : clé manquante '{key}'"
        actual_shape = tuple(bc_batch[key].shape)
        assert actual_shape == expected_shape, (
            f"BC['{key}'] shape {actual_shape} ≠ {expected_shape}"
        )

    # Dirichlet homogène : T* = 0 sur tout le bord
    assert torch.allclose(
        bc_batch["T_star"], torch.zeros_like(bc_batch["T_star"])
    ), "BC : T* doit être identiquement 0 (parois à T_amb)"

    assert bc_batch["x_star"].requires_grad, "BC : x_star doit avoir requires_grad=True"
    assert bc_batch["y_star"].requires_grad, "BC : y_star doit avoir requires_grad=True"
    assert bc_batch["t_star"].requires_grad, "BC : t_star doit avoir requires_grad=True"

    # Passage en NumPy 1-D pour les tests géométriques
    x_star = bc_batch["x_star"].detach().cpu().numpy().ravel()
    y_star = bc_batch["y_star"].detach().cpu().numpy().ravel()
    t_star = bc_batch["t_star"].detach()

    on_left = np.isclose(x_star, 0.0)
    on_right = np.isclose(x_star, 1.0)
    on_bottom = np.isclose(y_star, 0.0)
    on_top = np.isclose(y_star, 1.0)
    on_boundary = on_left | on_right | on_bottom | on_top
    assert on_boundary.all(), "BC : chaque point doit être sur le bord du domaine"

    # Domaine temporel de Fourier (le point sensible du bug fix)
    t_min = float(t_star.min())
    t_max = float(t_star.max())
    assert (t_star >= 0).all() and (t_star <= t_star_max + 1e-9).all(), (
        f"BC : t* doit vivre dans [0, t*_max={t_star_max}], "
        f"reçu [{t_min}, {t_max}]"
    )

    print(
        f"  [OK] BC  : N={n_bc}, T*=0, t*∈[{t_min:.4f}, {t_max:.4f}] "
        f"(t*_max={t_star_max:.4f}), "
        f"walls L/R/B/T = "
        f"{int(on_left.sum())}/{int(on_right.sum())}/"
        f"{int(on_bottom.sum())}/{int(on_top.sum())}"
    )


def validate_residual(res_batch: CollocationBatch, cfg: DataConfig) -> None:
    """
    Vérifie que le batch de collocation (résidu PDE) est bien formé.

    Invariants contrôlés
    --------------------
    1. Clés x_star, y_star, t_star + shapes (N_res, 1)
    2. requires_grad=True sur les 3 coordonnées (autograd du laplacien)
    3. (x*, y*) ∈ (0, 1)²     ← strictement intérieur spatial
    4. t* ∈ (0, t*_max)       ← strictement intérieur temporel de Fourier

    Parameters
    ----------
    res_batch : dict[str, Tensor]
        Sortie de sample_residual().
    cfg : DataConfig
        Fournit N_res et t_star_max.
    """
    n_res = cfg.N_res
    t_star_max = cfg.t_star_max
    expected_shape = (n_res, 1)

    for key in ("x_star", "y_star", "t_star"):
        assert key in res_batch, f"RES : clé manquante '{key}'"
        actual_shape = tuple(res_batch[key].shape)
        assert actual_shape == expected_shape, (
            f"RES['{key}'] shape {actual_shape} ≠ {expected_shape}"
        )
        assert res_batch[key].requires_grad, (
            f"RES : {key} doit avoir requires_grad=True (autograd PDE)"
        )

    x_star = res_batch["x_star"].detach()
    y_star = res_batch["y_star"].detach()
    t_star = res_batch["t_star"].detach()

    assert (x_star > 0).all() and (x_star < 1).all(), "RES : x* doit être dans (0, 1)"
    assert (y_star > 0).all() and (y_star < 1).all(), "RES : y* doit être dans (0, 1)"

    t_min = float(t_star.min())
    t_max = float(t_star.max())
    assert (t_star > 0).all() and (t_star < t_star_max).all(), (
        f"RES : t* doit vivre dans (0, t*_max={t_star_max}), "
        f"reçu [{t_min}, {t_max}]"
    )

    print(
        f"  [OK] RES : N={n_res}, (x*,y*)∈(0,1)², "
        f"t*∈[{t_min:.4f}, {t_max:.4f}] (t*_max={t_star_max:.4f})"
    )


# ==========================================================================
# Visualisation diagnostic
# ==========================================================================

def _draw_hot_object_outline(ax: Axes, cfg: DataConfig) -> None:
    """
    Superpose le contour de l'objet chaud sur un axe matplotlib spatial.

    POURQUOI tracer le contour ?
        Sur le scatter IC, les points T*=1 (rouges) doivent TOMBER
        à l'intérieur de ce contour. C'est la vérification visuelle
        que le masque géométrique est correct — très parlant en soutenance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axe spatial (x*, y*).
    cfg : DataConfig
        Géométrie de l'objet (forme, centre, rayon).
    """
    if cfg.obj_shape == "disk":
        outline = mpatches.Circle(
            (cfg.obj_cx, cfg.obj_cy),
            cfg.obj_radius,
            fill=False,
            edgecolor="#E45756",
            linewidth=2,
            linestyle="--",
            label=f"Objet chaud (disk r={cfg.obj_radius})",
        )
    else:
        side_length = 2.0 * cfg.obj_radius
        outline = mpatches.Rectangle(
            (cfg.obj_cx - cfg.obj_radius, cfg.obj_cy - cfg.obj_radius),
            side_length,
            side_length,
            fill=False,
            edgecolor="#E45756",
            linewidth=2,
            linestyle="--",
            label=f"Objet chaud (square r={cfg.obj_radius})",
        )
    ax.add_patch(outline)


def _tensor_to_numpy_1d(tensor: torch.Tensor) -> np.ndarray:
    """
    Convertit un tenseur PyTorch en vecteur NumPy 1-D pour matplotlib.

    Chaîne : detach (coupe autograd) → cpu (au cas où CUDA) → numpy → ravel.
    """
    return tensor.detach().cpu().numpy().ravel()


def plot_collocation_points(
    ic_batch: CollocationBatch,
    bc_batch: CollocationBatch,
    res_batch: CollocationBatch,
    cfg: DataConfig,
    out_path: Path,
) -> None:
    """
    Figure diagnostic 2×2 de l'Étape 1.

    Layout
    ------
        (0, 0) Nuage spatial (x*, y*)
               - résidu en bleu (sous-échantillonné pour lisibilité)
               - IC froid (T*=0) en vert, IC chaud (T*=1) en rouge
               - BC en orange sur le bord
               - contour de l'objet chaud en pointillés

        (0, 1) Histogramme de t*
               - BC et résidu bornés à [0, t*_max]
               - lignes verticales à t*=0 (IC) et t*=t*_max

        (1, 0) Barres : répartition des points BC par paroi
               (doit être ~équilibrée : N_bc/4 chacune)

        (1, 1) Cartouche texte : rappel de l'adimensionnement
               (utile en capture d'écran pour la soutenance)

    Parameters
    ----------
    ic_batch, bc_batch, res_batch : dict[str, Tensor]
        Les trois familles de points.
    cfg : DataConfig
        Pour les titres, bornes, géométrie objet.
    out_path : Path
        Chemin du PNG de sortie (ex. collocation_points.png).
    """
    # --- Préparation des tableaux NumPy --------------------------------
    # Sous-échantillon du résidu : 20 000 points rendraient le scatter illisible
    n_res_plot = min(4000, cfg.N_res)
    rng = np.random.default_rng(cfg.seed)
    res_plot_indices = rng.choice(cfg.N_res, size=n_res_plot, replace=False)

    ic_x = _tensor_to_numpy_1d(ic_batch["x_star"])
    ic_y = _tensor_to_numpy_1d(ic_batch["y_star"])
    ic_T = _tensor_to_numpy_1d(ic_batch["T_star"])

    bc_x = _tensor_to_numpy_1d(bc_batch["x_star"])
    bc_y = _tensor_to_numpy_1d(bc_batch["y_star"])
    bc_t = _tensor_to_numpy_1d(bc_batch["t_star"])
    bc_wall = _tensor_to_numpy_1d(bc_batch["wall"])

    res_x_all = _tensor_to_numpy_1d(res_batch["x_star"])
    res_y_all = _tensor_to_numpy_1d(res_batch["y_star"])
    res_t = _tensor_to_numpy_1d(res_batch["t_star"])
    res_x = res_x_all[res_plot_indices]
    res_y = res_y_all[res_plot_indices]

    # Masques IC chaud / froid (seuil 0.5 : T* ne vaut que 0 ou 1)
    is_cold = ic_T < 0.5
    is_hot = ~is_cold
    n_hot = int(is_hot.sum())

    # --- Figure ----------------------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle(
        "Étape 1 — Adimensionnement & Échantillonnage\n"
        r"$\partial T^*/\partial t^* = \partial^2 T^*/\partial x^{*2}"
        r" + \partial^2 T^*/\partial y^{*2}$"
        f"   |   $t^*\\in[0,\\,t^*_{{\\max}}={cfg.t_star_max:.4f}]$",
        fontsize=13,
        fontweight="bold",
    )

    # ---- (0, 0) Scatter spatial --------------------------------------
    ax_spatial: Axes = axes[0, 0]
    ax_spatial.scatter(
        res_x, res_y,
        s=2, c="#4C72B0", alpha=0.20,
        label=f"Résidu (n={n_res_plot}/{cfg.N_res})",
    )
    ax_spatial.scatter(
        ic_x[is_cold], ic_y[is_cold],
        s=8, c="#55A868", alpha=0.55,
        label=f"IC T*=0 (n={int(is_cold.sum())})",
    )
    ax_spatial.scatter(
        ic_x[is_hot], ic_y[is_hot],
        s=14, c="#E45756", alpha=0.9, zorder=5,
        label=f"IC T*=1 objet (n={n_hot})",
    )
    ax_spatial.scatter(
        bc_x, bc_y,
        s=10, c="#F58518", alpha=0.75,
        label=f"BC T*=0 (n={cfg.N_bc})",
    )
    _draw_hot_object_outline(ax_spatial, cfg)
    ax_spatial.set_xlabel(r"$x^*$")
    ax_spatial.set_ylabel(r"$y^*$")
    ax_spatial.set_xlim(-0.05, 1.05)
    ax_spatial.set_ylim(-0.05, 1.05)
    ax_spatial.set_aspect("equal")
    ax_spatial.set_title("Points de collocation (projection spatiale)")
    ax_spatial.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax_spatial.grid(True, alpha=0.3)

    # ---- (0, 1) Histogramme temporel ---------------------------------
    ax_time: Axes = axes[0, 1]
    t_star_max = cfg.t_star_max
    ax_time.hist(
        res_t, bins=40, color="#4C72B0", alpha=0.7,
        label="Résidu", density=True, range=(0.0, t_star_max),
    )
    ax_time.hist(
        bc_t, bins=40, color="#F58518", alpha=0.6,
        label="BC", density=True, range=(0.0, t_star_max),
    )
    ax_time.axvline(
        0.0, color="#55A868", linewidth=2, linestyle="--",
        label="IC (t*=0)",
    )
    ax_time.axvline(
        t_star_max, color="#E45756", linewidth=2, linestyle=":",
        label=fr"$t^*_{{\max}}={t_star_max:.4f}$",
    )
    ax_time.set_xlabel(r"$t^*$")
    ax_time.set_ylabel("densité")
    ax_time.set_xlim(0.0, t_star_max * 1.05)
    ax_time.set_title(
        fr"Distribution temporelle  —  $t^*\in[0,\ {t_star_max:.4f}]$"
    )
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)

    # ---- (1, 0) Répartition par paroi --------------------------------
    ax_walls: Axes = axes[1, 0]
    wall_names = [
        "Gauche\n(x*=0)",
        "Droite\n(x*=1)",
        "Bas\n(y*=0)",
        "Haut\n(y*=1)",
    ]
    wall_counts = [int((bc_wall == wall_id).sum()) for wall_id in range(4)]
    wall_colors = ["#E07A5F", "#3D405B", "#81B29A", "#F2CC8F"]
    bars = ax_walls.bar(
        wall_names, wall_counts,
        color=wall_colors, edgecolor="black", linewidth=0.5,
    )
    ax_walls.set_ylabel("nombre de points")
    ax_walls.set_title("Répartition BC par paroi")
    y_max = max(wall_counts) * 1.2 if wall_counts else 1.0
    ax_walls.set_ylim(0, y_max)
    for bar, count in zip(bars, wall_counts):
        ax_walls.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(wall_counts) * 0.02,
            str(count),
            ha="center", va="bottom", fontsize=9,
        )
    ax_walls.grid(True, axis="y", alpha=0.3)

    # ---- (1, 1) Cartouche adimensionnement ---------------------------
    ax_text: Axes = axes[1, 1]
    ax_text.axis("off")
    summary_text = (
        "Adimensionnement\n"
        "─────────────────────────────\n"
        f"  x* = x / Lx             Lx = {cfg.Lx} m\n"
        f"  y* = y / Ly             Ly = {cfg.Ly} m\n"
        f"  t* = α t / L²           t*_max = {cfg.t_star_max:.4f}\n"
        f"  T* = (T − T_amb)/ΔT     ΔT = {cfg.delta_T}°C\n"
        "\n"
        "Physique\n"
        "─────────────────────────────\n"
        f"  α  = {cfg.alpha:.2e} m²/s\n"
        f"  T_amb = {cfg.T_amb}°C   T_obj = {cfg.T_obj}°C\n"
        f"  t_ref = L²/α = {cfg.t_ref:.1f} s\n"
        f"  t_max = {cfg.t_max} s\n"
        f"  Objet : {cfg.obj_shape} r={cfg.obj_radius} @ "
        f"({cfg.obj_cx},{cfg.obj_cy})\n"
        "\n"
        "Échantillonnage\n"
        "─────────────────────────────\n"
        f"  N_ic  = {cfg.N_ic:>6d}   t*=0, T*∈{{0,1}} (hot={n_hot})\n"
        f"  N_bc  = {cfg.N_bc:>6d}   parois, T*=0, t*≤t*_max\n"
        f"  N_res = {cfg.N_res:>6d}   intérieur, t*∈(0,t*_max)\n"
        f"  seed  = {cfg.seed}\n"
        "\n"
        "Équation cible (adim.)\n"
        "─────────────────────────────\n"
        "  ∂T*/∂t* = ∂²T*/∂x*² + ∂²T*/∂y*²\n"
    )
    ax_text.text(
        0.05, 0.95, summary_text,
        transform=ax_text.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    # --- Sauvegarde ----------------------------------------------------
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  [OK] Figure sauvegardée → {out_path.resolve()}")


# ==========================================================================
# Point d'entrée CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse les arguments en ligne de commande.

    Returns
    -------
    argparse.Namespace
        method, seed, n_ic, n_bc, n_res, out, device
    """
    parser = argparse.ArgumentParser(
        description="PINN Thermal 2D — Étape 1 : adimensionnement & échantillonnage"
    )
    parser.add_argument(
        "--method",
        choices=["sobol", "lhs", "uniform"],
        default="sobol",
        help="Stratégie d'échantillonnage quasi-aléatoire (défaut: sobol)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Graine de reproductibilité (défaut: 42)",
    )
    parser.add_argument(
        "--n-ic", type=int, default=None,
        help="Surcharge de N_ic (défaut: DataConfig.N_ic=2000)",
    )
    parser.add_argument(
        "--n-bc", type=int, default=None,
        help="Surcharge de N_bc (défaut: DataConfig.N_bc=2000)",
    )
    parser.add_argument(
        "--n-res", type=int, default=None,
        help="Surcharge de N_res (défaut: DataConfig.N_res=20000)",
    )
    parser.add_argument(
        "--out", type=str, default="collocation_points.png",
        help="Chemin de la figure PNG de sortie",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device PyTorch : 'cpu' ou 'cuda'",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> DataConfig:
    """
    Construit un DataConfig à partir des arguments CLI.

    Seuls les champs explicitement fournis (non-None) surchargent
    les valeurs par défaut de la dataclass. Cela permet de lancer
    `python main_step1.py` sans aucun flag et d'obtenir la config nominale.
    """
    config_kwargs = {
        "seed": args.seed,
        "device": args.device,
    }
    if args.n_ic is not None:
        config_kwargs["N_ic"] = args.n_ic
    if args.n_bc is not None:
        config_kwargs["N_bc"] = args.n_bc
    if args.n_res is not None:
        config_kwargs["N_res"] = args.n_res
    return DataConfig(**config_kwargs)


def print_tensor_summary(
    name: str,
    batch: CollocationBatch,
) -> None:
    """
    Affiche shape / dtype / requires_grad de chaque tenseur d'un batch.

    Très utile en soutenance pour montrer d'un coup d'œil que :
        - les shapes sont bien (N, 1)
        - requires_grad est True sur les coordonnées, False sur T*
    """
    print(f"  {name}:")
    for key, tensor in batch.items():
        requires_grad = getattr(tensor, "requires_grad", False)
        print(
            f"    {key:8s} shape={tuple(tensor.shape)}  "
            f"dtype={tensor.dtype}  requires_grad={requires_grad}"
        )


def main() -> int:
    """
    Orchestration de l'Étape 1.

    Returns
    -------
    int
        Code retour shell (0 = succès).
    """
    args = parse_args()
    cfg = build_config_from_args(args)
    set_seed(cfg.seed)

    # ----- Bannière configuration --------------------------------------
    print(cfg.summary())
    print(f"\nMéthode d'échantillonnage : {args.method}")

    # ----- 1/4 Génération ----------------------------------------------
    print("\n[1/4] Génération des points…")
    ic_batch = sample_ic(cfg, method=args.method)
    bc_batch = sample_bc(cfg, method=args.method)
    res_batch = sample_residual(cfg, method=args.method)

    # ----- 2/4 Validation ----------------------------------------------
    print("[2/4] Validation des invariants…")
    validate_ic(ic_batch, cfg)
    validate_bc(bc_batch, cfg)
    validate_residual(res_batch, cfg)

    # ----- 3/4 Figure --------------------------------------------------
    print("[3/4] Export de la figure diagnostic…")
    plot_collocation_points(
        ic_batch, bc_batch, res_batch, cfg, out_path=Path(args.out)
    )

    # ----- 4/4 Récap tenseurs ------------------------------------------
    print("[4/4] Récapitulatif des tenseurs :")
    print_tensor_summary("IC", ic_batch)
    print_tensor_summary("BC", bc_batch)
    print_tensor_summary("RES", res_batch)

    # Confirmation explicite des bornes (reprise des logs de validation,
    # regroupée pour la capture d'écran soutenance).
    bc_t = bc_batch["t_star"].detach()
    res_t = res_batch["t_star"].detach()
    ic_T = ic_batch["T_star"].detach()
    print("\n[confirm] Domaines temporels / température :")
    print(
        f"  BC  t* ∈ [{float(bc_t.min()):.6f}, {float(bc_t.max()):.6f}]  "
        f"(attendu [0, {cfg.t_star_max:.4f}])"
    )
    print(
        f"  RES t* ∈ [{float(res_t.min()):.6f}, {float(res_t.max()):.6f}]  "
        f"(attendu (0, {cfg.t_star_max:.4f}))"
    )
    print(
        f"  IC  T* ∈ [{float(ic_T.min()):.2f}, {float(ic_T.max()):.2f}]  "
        f"(attendu [0.00, 1.00])"
    )

    print("\n✓ Étape 1 terminée avec succès.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
