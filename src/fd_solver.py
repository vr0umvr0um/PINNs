"""
==============================================================================
src/fd_solver.py — Solveur de référence par différences finies (Étape 4)
==============================================================================

POURQUOI UN SOLVEUR CLASSIQUE ?
-------------------------------
Un PINN n'a AUCUNE vérité terrain interne : il minimise un résidu. Rien ne
dit que la solution trouvée est la bonne. Pour mesurer son erreur il faut
une solution de RÉFÉRENCE obtenue par une méthode indépendante, déjà
validée. Ici : un schéma explicite aux différences finies (FTCS — Forward
Time, Centered Space) sur l'équation adimensionnée du projet :

    ∂T*/∂t* = ∂²T*/∂x*² + ∂²T*/∂y*²        (x*, y*) ∈ [0, 1]², t* ∈ [0, t*_max]

avec les MÊMES conditions que le PINN :
    IC : T* = 1 dans l'objet chaud, 0 ailleurs
    BC : T* = 0 sur les 4 parois (Dirichlet)

LE SCHÉMA
---------
Grille régulière de n × n nœuds, pas h = 1/(n-1). Pour un nœud intérieur :

    T^{k+1}_{ij} = T^k_{ij} + r · ( T^k_{i+1,j} + T^k_{i-1,j}
                                   + T^k_{i,j+1} + T^k_{i,j-1} − 4 T^k_{ij} )

avec r = Δt* / h². Le schéma est d'ordre 1 en temps et 2 en espace.

CONDITION DE STABILITÉ (CFL diffusive)
--------------------------------------
Le schéma explicite n'est stable que si  r ≤ 1/4  en 2D. Au-delà, les modes
haute fréquence sont amplifiés à chaque pas et la solution explose. On
impose donc r ≤ `STABILITY_RATIO` = 0.2 (marge de sécurité). Le pas de temps
est ensuite ajusté pour tomber EXACTEMENT sur chaque instant demandé : on
compare PINN et référence aux mêmes t*, sans interpolation temporelle.

LIMITE ASSUMÉE
--------------
La condition initiale est un créneau (discontinuité). La grille ne peut
pas la représenter exactement : un nœud est « chaud » ou « froid ». L'erreur
de discrétisation du créneau décroît en O(h) ; l'effet est négligeable dès
que le profil a diffusé (t* > quelques h²).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from config import DataConfig
from src.utils import mask_hot_object

# r = Δt*/h² maximal autorisé. Limite théorique : 0.25 (2D). Marge : 20 %.
STABILITY_RATIO: float = 0.2


@dataclass
class FDSolution:
    """
    Solution de référence échantillonnée sur une grille régulière.

    Attributs
    ---------
    x_star, y_star : np.ndarray de shape (n,)
        Coordonnées des nœuds ∈ [0, 1].
    t_star : np.ndarray de shape (n_times,)
        Instants des instantanés, de 0 à t*_max.
    T_star : np.ndarray de shape (n_times, n, n)
        Champ T*[k, i, j] = T*(x_i, y_j, t_k). Convention : le premier
        indice spatial est x, le second y (indexing="ij").
    n_steps : int
        Nombre total de pas de temps explicites effectués.
    """

    x_star: np.ndarray
    y_star: np.ndarray
    t_star: np.ndarray
    T_star: np.ndarray
    n_steps: int

    @property
    def n_grid(self) -> int:
        return int(self.x_star.shape[0])

    def at(self, t_star: float) -> np.ndarray:
        """Champ (n, n) à l'instant enregistré le plus proche de t_star."""
        index = int(np.argmin(np.abs(self.t_star - t_star)))
        return self.T_star[index]


def initial_field(
    x_star: np.ndarray,
    y_star: np.ndarray,
    cfg: DataConfig,
) -> np.ndarray:
    """
    Condition initiale sur la grille : 1 dans l'objet chaud, 0 ailleurs,
    0 imposé sur les parois (Dirichlet, même à t* = 0).

    Returns
    -------
    np.ndarray of shape (n_x, n_y) — indexing "ij".
    """
    grid_x, grid_y = np.meshgrid(x_star, y_star, indexing="ij")
    inside = mask_hot_object(
        grid_x,
        grid_y,
        center_x=cfg.obj_cx,
        center_y=cfg.obj_cy,
        radius=cfg.obj_radius,
        shape=cfg.obj_shape,
    )
    field = inside.astype(np.float64)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
    return field


def solve_heat_fd(
    cfg: DataConfig,
    n_grid: int = 101,
    n_times: int = 51,
    t_star_values: Optional[Sequence[float]] = None,
    initial_condition: Optional[np.ndarray] = None,
) -> FDSolution:
    """
    Résout l'équation de la chaleur adimensionnée par différences finies.

    Parameters
    ----------
    cfg : DataConfig
        Fournit l'objet chaud et l'horizon t*_max.
    n_grid : int
        Nombre de nœuds par direction (≥ 5). 101 → h = 0.01.
    n_times : int
        Nombre d'instantanés régulièrement répartis sur [0, t*_max]
        (ignoré si `t_star_values` est fourni). Inclut t* = 0 et t*_max.
    t_star_values : sequence of float, optional
        Instants explicites (triés, ≥ 0). Remplace la grille régulière.
    initial_condition : np.ndarray of shape (n_grid, n_grid), optional
        Remplace le créneau de l'objet (sert aux tests de convergence avec
        un mode propre exact). Les parois sont de toute façon remises à 0.

    Returns
    -------
    FDSolution
    """
    if n_grid < 5:
        raise ValueError(f"n_grid doit être ≥ 5, reçu {n_grid}.")

    if t_star_values is None:
        if n_times < 2:
            raise ValueError(f"n_times doit être ≥ 2, reçu {n_times}.")
        times = np.linspace(0.0, cfg.t_star_max, n_times)
    else:
        times = np.asarray(t_star_values, dtype=np.float64)
        if times.ndim != 1 or times.size < 1 or np.any(np.diff(times) < 0) or times[0] < 0:
            raise ValueError("t_star_values doit être 1-D, trié et ≥ 0.")

    nodes = np.linspace(0.0, 1.0, n_grid)
    step = nodes[1] - nodes[0]

    if initial_condition is None:
        field = initial_field(nodes, nodes, cfg)
    else:
        field = np.array(initial_condition, dtype=np.float64, copy=True)
        if field.shape != (n_grid, n_grid):
            raise ValueError(
                f"initial_condition doit avoir la shape {(n_grid, n_grid)}, "
                f"reçu {field.shape}."
            )
        field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0

    snapshots = np.empty((times.size, n_grid, n_grid), dtype=np.float64)
    current_time = 0.0
    total_steps = 0
    dt_max = STABILITY_RATIO * step * step

    for index, target_time in enumerate(times):
        interval = target_time - current_time
        if interval > 0.0:
            # On découpe l'intervalle en n_sub pas égaux, chacun ≤ dt_max :
            # le dernier pas tombe exactement sur target_time.
            n_sub = int(np.ceil(interval / dt_max))
            ratio = interval / n_sub / (step * step)
            for _ in range(n_sub):
                laplacian_sum = (
                    field[2:, 1:-1] + field[:-2, 1:-1]
                    + field[1:-1, 2:] + field[1:-1, :-2]
                    - 4.0 * field[1:-1, 1:-1]
                )
                field[1:-1, 1:-1] += ratio * laplacian_sum
            total_steps += n_sub
            current_time = target_time
        snapshots[index] = field

    return FDSolution(
        x_star=nodes,
        y_star=nodes.copy(),
        t_star=times,
        T_star=snapshots,
        n_steps=total_steps,
    )


__all__ = ["FDSolution", "STABILITY_RATIO", "initial_field", "solve_heat_fd"]
