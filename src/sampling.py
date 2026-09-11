"""
Échantillonnage des points de collocation pour le PINN thermique 2D.

Trois familles de points (domaine adimensionné) :

1. Condition initiale  (sample_ic)       — t* = 0, (x*, y*) ∈ [0, 1]²
2. Conditions aux limites (sample_bc)    — 4 parois, T* = 0 (Dirichlet ambiante)
3. Résidu PDE (sample_residual)          — (x*, y*, t*) ∈ ]0, 1[³

Chaque fonction retourne un dictionnaire de tenseurs PyTorch. Les
coordonnées spatiales / temporelles sont créées avec requires_grad=True
afin de permettre le calcul automatique des dérivées (autograd) lors
de l'évaluation du résidu.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import numpy as np
import torch

from config import DEFAULT_CONFIG, DataConfig
from src.utils import latin_hypercube, set_seed, sobol_sample, to_tensor

SamplingMethod = Literal["sobol", "lhs", "uniform"]


def _sample_unit(
    n: int,
    dim: int,
    method: SamplingMethod,
    seed: int,
) -> np.ndarray:
    """Tire n points dans [0, 1]^dim selon la méthode choisie."""
    if method == "sobol":
        return sobol_sample(n, dim, seed=seed)
    if method == "lhs":
        return latin_hypercube(n, dim, seed=seed)
    if method == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random((n, dim), dtype=np.float64)
    raise ValueError(f"Unknown sampling method: {method!r}")


def sample_ic(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    T_star_init: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Échantillonne les points de condition initiale.

    À t* = 0, le champ de température adimensionné vaut T*_init
    (par défaut 0 ⇒ pièce à T_amb partout).

    Parameters
    ----------
    cfg : DataConfig, optional
        Configuration (N_ic, seed, device…).
    method : {'sobol', 'lhs', 'uniform'}
        Stratégie d'échantillonnage spatial.
    T_star_init : float
        Valeur de T* imposée sur l'IC (défaut 0.0).

    Returns
    -------
    dict with keys
        x_star : (N_ic, 1)  requires_grad=True
        y_star : (N_ic, 1)  requires_grad=True
        t_star : (N_ic, 1)  requires_grad=True  (tout à 0)
        T_star : (N_ic, 1)  requires_grad=False (cible IC)
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    xy = _sample_unit(cfg.N_ic, dim=2, method=method, seed=cfg.seed)
    x = xy[:, 0]
    y = xy[:, 1]
    t = np.zeros(cfg.N_ic, dtype=np.float64)
    T = np.full(cfg.N_ic, T_star_init, dtype=np.float64)

    device = cfg.device
    return {
        "x_star": to_tensor(x, requires_grad=True, device=device),
        "y_star": to_tensor(y, requires_grad=True, device=device),
        "t_star": to_tensor(t, requires_grad=True, device=device),
        "T_star": to_tensor(T, requires_grad=False, device=device),
    }


def sample_bc(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    T_star_bc: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Échantillonne les points de conditions aux limites Dirichlet.

    Les 4 parois du carré [0, 1]² sont peuplées équitablement
    (N_bc // 4 points chacune, le reste réparti sur les premières parois).
    Sur chaque paroi : T* = T_star_bc (défaut 0 ⇒ T_amb).
    Le temps t* est échantillonné uniformément sur [0, 1].

    Parois :
        0 gauche  (x* = 0, y* libre)
        1 droite  (x* = 1, y* libre)
        2 bas     (y* = 0, x* libre)
        3 haut    (y* = 1, x* libre)

    Parameters
    ----------
    cfg : DataConfig, optional
    method : {'sobol', 'lhs', 'uniform'}
        Utilisé pour (coordonnée libre, t*).
    T_star_bc : float
        Température adimensionnée imposée au bord.

    Returns
    -------
    dict with keys
        x_star, y_star, t_star : (N_bc, 1) requires_grad=True
        T_star                 : (N_bc, 1) requires_grad=False
        wall                   : (N_bc, 1) int64, index de paroi {0,1,2,3}
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    n = cfg.N_bc
    n_per_wall = n // 4
    remainder = n - 4 * n_per_wall
    counts = [n_per_wall + (1 if i < remainder else 0) for i in range(4)]

    xs, ys, ts, walls = [], [], [], []
    # Seeds décalées par paroi pour diversifier les sous-échantillons
    for wall_id, n_wall in enumerate(counts):
        if n_wall == 0:
            continue
        # 2D sample : coordonnée libre + temps
        free_t = _sample_unit(
            n_wall, dim=2, method=method, seed=cfg.seed + wall_id + 1
        )
        free_coord = free_t[:, 0]
        t_coord = free_t[:, 1]

        if wall_id == 0:  # gauche x*=0
            x_coord = np.zeros(n_wall)
            y_coord = free_coord
        elif wall_id == 1:  # droite x*=1
            x_coord = np.ones(n_wall)
            y_coord = free_coord
        elif wall_id == 2:  # bas y*=0
            x_coord = free_coord
            y_coord = np.zeros(n_wall)
        else:  # haut y*=1
            x_coord = free_coord
            y_coord = np.ones(n_wall)

        xs.append(x_coord)
        ys.append(y_coord)
        ts.append(t_coord)
        walls.append(np.full(n_wall, wall_id, dtype=np.int64))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    t = np.concatenate(ts)
    wall = np.concatenate(walls)
    T = np.full(n, T_star_bc, dtype=np.float64)

    # Mélange pour éviter un ordre paroi-par-paroi trop structuré
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)
    x, y, t, wall, T = x[perm], y[perm], t[perm], wall[perm], T[perm]

    device = cfg.device
    return {
        "x_star": to_tensor(x, requires_grad=True, device=device),
        "y_star": to_tensor(y, requires_grad=True, device=device),
        "t_star": to_tensor(t, requires_grad=True, device=device),
        "T_star": to_tensor(T, requires_grad=False, device=device),
        "wall": torch.as_tensor(wall, dtype=torch.int64, device=device).unsqueeze(-1),
    }


def sample_residual(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    interior_eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Échantillonne les points de collocation pour le résidu PDE.

    Points dans le cube ouvert ]eps, 1-eps[³ afin d'éviter le double
    comptage avec IC (t*=0) et BC (bords spatiaux).

    L'équation adimensionnée à résidualiser (Étapes suivantes) est :

        r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)

    Parameters
    ----------
    cfg : DataConfig, optional
    method : {'sobol', 'lhs', 'uniform'}
    interior_eps : float
        Marge pour rester strictement intérieur.

    Returns
    -------
    dict with keys
        x_star, y_star, t_star : (N_res, 1) requires_grad=True
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    raw = _sample_unit(cfg.N_res, dim=3, method=method, seed=cfg.seed + 99)
    # Affine de [0, 1] → [eps, 1-eps]
    lo, hi = interior_eps, 1.0 - interior_eps
    scaled = lo + (hi - lo) * raw

    x = scaled[:, 0]
    y = scaled[:, 1]
    t = scaled[:, 2]

    device = cfg.device
    return {
        "x_star": to_tensor(x, requires_grad=True, device=device),
        "y_star": to_tensor(y, requires_grad=True, device=device),
        "t_star": to_tensor(t, requires_grad=True, device=device),
    }


def sample_all(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Raccourci : génère IC + BC + résidu d'un coup."""
    cfg = cfg or DEFAULT_CONFIG
    return {
        "ic": sample_ic(cfg, method=method),
        "bc": sample_bc(cfg, method=method),
        "res": sample_residual(cfg, method=method),
    }


__all__ = [
    "sample_ic",
    "sample_bc",
    "sample_residual",
    "sample_all",
]
