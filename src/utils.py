"""
Utilitaires d'échantillonnage quasi-aléatoire et masques géométriques.

Fournit :
    - sobol_sample      : séquence de Sobol (faible discrépance)
    - latin_hypercube   : Latin Hypercube Sampling (LHS)
    - mask_interior     : booléen points strictement intérieurs au carré [0,1]²
    - mask_boundary     : booléen points sur le bord du carré [0,1]²
    - set_seed          : reproductibilité NumPy / PyTorch
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import qmc


def set_seed(seed: int = 42) -> None:
    """Fixe les graines NumPy et PyTorch pour la reproductibilité."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sobol_sample(
    n: int,
    dim: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
    scramble: bool = True,
) -> np.ndarray:
    """
    Échantillonne `n` points dans [0, 1]^dim via une séquence de Sobol,
    puis les affine éventuellement dans `bounds`.

    Parameters
    ----------
    n : int
        Nombre de points.
    dim : int
        Dimension de l'espace.
    bounds : ndarray of shape (dim, 2), optional
        Bornes (min, max) par dimension. Si None, reste dans [0, 1]^dim.
    seed : int
        Graine du scramble.
    scramble : bool
        Active le scramble Owen (recommandé).

    Returns
    -------
    samples : ndarray of shape (n, dim)
    """
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)

    engine = qmc.Sobol(d=dim, scramble=scramble, seed=seed)
    # Sobol exige souvent une puissance de 2 pour l'équilibre ; on tire
    # le prochain power-of-2 puis on sous-échantillonne si besoin.
    n_pow2 = 1 << int(np.ceil(np.log2(max(n, 1))))
    raw = engine.random(n_pow2)[:n]

    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (dim, 2):
            raise ValueError(f"bounds must have shape ({dim}, 2), got {bounds.shape}")
        raw = qmc.scale(raw, bounds[:, 0], bounds[:, 1])

    return raw.astype(np.float64)


def latin_hypercube(
    n: int,
    dim: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Latin Hypercube Sampling dans [0, 1]^dim (puis scaling optionnel).

    Parameters
    ----------
    n : int
        Nombre de points.
    dim : int
        Dimension.
    bounds : ndarray of shape (dim, 2), optional
        Bornes (min, max) par dimension.
    seed : int
        Graine.

    Returns
    -------
    samples : ndarray of shape (n, dim)
    """
    if n <= 0:
        return np.zeros((0, dim), dtype=np.float64)

    engine = qmc.LatinHypercube(d=dim, seed=seed)
    raw = engine.random(n)

    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        if bounds.shape != (dim, 2):
            raise ValueError(f"bounds must have shape ({dim}, 2), got {bounds.shape}")
        raw = qmc.scale(raw, bounds[:, 0], bounds[:, 1])

    return raw.astype(np.float64)


def mask_interior(
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """
    Masque booléen des points strictement intérieurs au carré unité [0, 1]².

    Parameters
    ----------
    x, y : ndarray
        Coordonnées (même shape).
    tol : float
        Tolérance numérique au bord.

    Returns
    -------
    mask : ndarray of bool
    """
    return (x > tol) & (x < 1.0 - tol) & (y > tol) & (y < 1.0 - tol)


def mask_boundary(
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Masque booléen des points situés sur le bord du carré unité [0, 1]².

    Un point est « au bord » si au moins une coordonnée vaut 0 ou 1
    (à `tol` près) et les deux restent dans [0, 1].

    Parameters
    ----------
    x, y : ndarray
        Coordonnées (même shape).
    tol : float
        Tolérance.

    Returns
    -------
    mask : ndarray of bool
    """
    on_x = (np.abs(x - 0.0) <= tol) | (np.abs(x - 1.0) <= tol)
    on_y = (np.abs(y - 0.0) <= tol) | (np.abs(y - 1.0) <= tol)
    in_box = (
        (x >= -tol) & (x <= 1.0 + tol) & (y >= -tol) & (y <= 1.0 + tol)
    )
    return in_box & (on_x | on_y)


def which_boundary(
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Identifie la paroi de chaque point bord.

    Codes :
        0 = gauche  (x* = 0)
        1 = droite  (x* = 1)
        2 = bas     (y* = 0)
        3 = haut    (y* = 1)
       -1 = non-bord / coin ambigu (priorité x si coin)

    Returns
    -------
    labels : ndarray of int, same shape as x
    """
    labels = np.full(np.broadcast(x, y).shape, -1, dtype=np.int64)
    # Priorité : gauche, droite, bas, haut
    labels = np.where(np.abs(x - 0.0) <= tol, 0, labels)
    labels = np.where(np.abs(x - 1.0) <= tol, 1, labels)
    # y-walls only if not already tagged as x-wall (corners keep x-tag)
    not_x = labels < 0
    labels = np.where(not_x & (np.abs(y - 0.0) <= tol), 2, labels)
    labels = np.where(not_x & (np.abs(y - 1.0) <= tol), 3, labels)
    return labels


def to_tensor(
    array: np.ndarray,
    requires_grad: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convertit un ndarray en tenseur PyTorch colonne (N, 1) si 1-D."""
    t = torch.as_tensor(array, dtype=dtype, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    t = t.requires_grad_(requires_grad)
    return t


def unit_cube_bounds(dim: int) -> np.ndarray:
    """Bornes [[0, 1]] * dim."""
    return np.tile(np.array([[0.0, 1.0]]), (dim, 1))


__all__ = [
    "set_seed",
    "sobol_sample",
    "latin_hypercube",
    "mask_interior",
    "mask_boundary",
    "which_boundary",
    "to_tensor",
    "unit_cube_bounds",
]
