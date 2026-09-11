"""
Échantillonnage des points de collocation pour le PINN thermique 2D.

Trois familles de points (domaine adimensionné) :

1. Condition initiale  (sample_ic)
       t* = 0, (x*, y*) ∈ [0, 1]²
       T* = 1 à l'intérieur de l'objet chaud, T* = 0 à l'extérieur

2. Conditions aux limites (sample_bc)
       4 parois, T* = 0 (Dirichlet ambiante)
       t* ∈ [0, t*_max]   (t*_max = α t_max / L_ref² ≈ 0.1)

3. Résidu PDE (sample_residual)
       (x*, y*) ∈ ]0, 1[²,  t* ∈ ]0, t*_max[

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
from src.utils import latin_hypercube, mask_hot_object, set_seed, sobol_sample, to_tensor

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


def _scale_time(u: np.ndarray, t_max_star: float) -> np.ndarray:
    """Affine u ∈ [0, 1] → t* ∈ [0, t*_max]."""
    return u * t_max_star


def _ic_temperature(
    x: np.ndarray,
    y: np.ndarray,
    cfg: DataConfig,
) -> np.ndarray:
    """
    Champ T* de condition initiale.

    T* = 1.0 à l'intérieur de l'objet chaud, 0.0 à l'extérieur.
    """
    inside = mask_hot_object(
        x,
        y,
        cx=cfg.obj_cx,
        cy=cfg.obj_cy,
        radius=cfg.obj_radius,
        shape=cfg.obj_shape,
    )
    T = np.zeros(x.shape[0], dtype=np.float64)
    T[inside] = 1.0
    return T


def sample_ic(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    T_star_init: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Échantillonne les points de condition initiale.

    À t* = 0 :
        - T* = 1.0  à l'intérieur de l'objet chaud (disque/carré)
        - T* = 0.0  à l'extérieur (pièce à T_amb)

    Si `T_star_init` est fourni (float), ce champ uniforme écrase le
    masque objet (utile pour des tests unitaires ciblés).

    Parameters
    ----------
    cfg : DataConfig, optional
        Configuration (N_ic, seed, device, géométrie objet…).
    method : {'sobol', 'lhs', 'uniform'}
        Stratégie d'échantillonnage spatial.
    T_star_init : float, optional
        Si donné, impose une IC uniforme (ignore l'objet chaud).

    Returns
    -------
    dict with keys
        x_star : (N_ic, 1)  requires_grad=True
        y_star : (N_ic, 1)  requires_grad=True
        t_star : (N_ic, 1)  requires_grad=True  (tout à 0)
        T_star : (N_ic, 1)  requires_grad=False (cible IC ∈ {0, 1})
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    xy = _sample_unit(cfg.N_ic, dim=2, method=method, seed=cfg.seed)
    x = xy[:, 0]
    y = xy[:, 1]
    t = np.zeros(cfg.N_ic, dtype=np.float64)

    if T_star_init is not None:
        T = np.full(cfg.N_ic, float(T_star_init), dtype=np.float64)
    else:
        T = _ic_temperature(x, y, cfg)

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

    Le temps t* est échantillonné dans **[0, t*_max]**, avec
    t*_max = α t_max / L_ref² (≈ 0.1 pour la config par défaut).

    Parois :
        0 gauche  (x* = 0, y* libre)
        1 droite  (x* = 1, y* libre)
        2 bas     (y* = 0, x* libre)
        3 haut    (y* = 1, x* libre)

    Parameters
    ----------
    cfg : DataConfig, optional
    method : {'sobol', 'lhs', 'uniform'}
        Utilisé pour (coordonnée libre, t*_unit).
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
    t_max_star = cfg.t_star_max

    xs, ys, ts, walls = [], [], [], []
    # Seeds décalées par paroi pour diversifier les sous-échantillons
    for wall_id, n_wall in enumerate(counts):
        if n_wall == 0:
            continue
        # 2D sample : coordonnée libre + temps unitaire
        free_t = _sample_unit(
            n_wall, dim=2, method=method, seed=cfg.seed + wall_id + 1
        )
        free_coord = free_t[:, 0]
        t_coord = _scale_time(free_t[:, 1], t_max_star)  # → [0, t*_max]

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

    - (x*, y*) ∈ ]eps, 1-eps[²  (strictement intérieur spatial)
    - t*      ∈ ]eps_t, t*_max - eps_t[  avec t*_max = α t_max / L_ref²

    L'équation adimensionnée à résidualiser (Étapes suivantes) est :

        r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)

    Parameters
    ----------
    cfg : DataConfig, optional
    method : {'sobol', 'lhs', 'uniform'}
    interior_eps : float
        Marge relative pour rester strictement intérieur (appliquée
        spatialement sur [0,1] et temporellement sur [0, t*_max]).

    Returns
    -------
    dict with keys
        x_star, y_star, t_star : (N_res, 1) requires_grad=True
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    raw = _sample_unit(cfg.N_res, dim=3, method=method, seed=cfg.seed + 99)
    t_max_star = cfg.t_star_max

    # Spatial : [0, 1] → [eps, 1-eps]
    lo_s, hi_s = interior_eps, 1.0 - interior_eps
    x = lo_s + (hi_s - lo_s) * raw[:, 0]
    y = lo_s + (hi_s - lo_s) * raw[:, 1]

    # Temporel : [0, 1] → [eps_t, t*_max - eps_t]
    # eps_t proportionnel pour rester cohérent quel que soit t*_max
    eps_t = interior_eps * t_max_star
    lo_t, hi_t = eps_t, t_max_star - eps_t
    t = lo_t + (hi_t - lo_t) * raw[:, 2]

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
