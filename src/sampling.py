"""
==============================================================================
src/sampling.py — Génération des points de collocation du PINN
==============================================================================

POURQUOI TROIS FAMILLES DE POINTS ?
-----------------------------------
Un PINN minimise une loss totale :

    L = λ_ic · L_ic  +  λ_bc · L_bc  +  λ_res · L_res

chacune évaluée sur SA propre famille de points :

1. sample_ic        → condition initiale
       t* = 0
       (x*, y*) ∈ [0, 1]²
       T* = 1 dans l'objet chaud, 0 ailleurs
       Loss typique : MSE( T_pred(x*,y*,0), T*_cible )

2. sample_bc        → conditions aux limites Dirichlet
       4 parois du carré
       T* = 0  (parois maintenues à T_amb)
       t* ∈ [0, t*_max]
       Loss typique : MSE( T_pred(paroi, t*), 0 )

3. sample_residual  → résidu de l'équation aux dérivées partielles
       (x*, y*) ∈ (0, 1)²   (strictement intérieur)
       t*      ∈ (0, t*_max)
       Loss typique : MSE( ∂T*/∂t* − Δ*T* ,  0 )
       Les dérivées sont obtenues par AUTOGRAD PyTorch
       (d'où requires_grad=True sur x*, y*, t*).

Convention de sortie
--------------------
Chaque fonction renvoie un dict[str, torch.Tensor] :

    {
      "x_star": Tensor de shape (N, 1), requires_grad=True,
      "y_star": Tensor de shape (N, 1), requires_grad=True,
      "t_star": Tensor de shape (N, 1), requires_grad=True,
      "T_star": Tensor de shape (N, 1), requires_grad=False,  # si cible
      ...
    }

La shape (N, 1) = (batch_size, n_features=1) est le format attendu
par un MLP qui prend une coordonnée scalaire à la fois, ou qui les
concatène en (N, 3) juste avant l'avant.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import numpy as np
import torch

from config import DEFAULT_CONFIG, DataConfig
from src.utils import (
    latin_hypercube,
    mask_hot_object,
    set_seed,
    sobol_sample,
    to_tensor,
)

# Méthodes d'échantillonnage supportées.
# "sobol"    : faible discrépance (recommandé par défaut)
# "lhs"      : Latin Hypercube
# "uniform"  : i.i.d. uniforme (baseline, moins bon en couverture)
SamplingMethod = Literal["sobol", "lhs", "uniform"]

# Type de retour standardisé pour IC / BC / résidu
CollocationBatch = Dict[str, torch.Tensor]


# ==========================================================================
# Helpers privés (préfix _  →  non exportés)
# ==========================================================================

def _sample_in_unit_hypercube(
    n_points: int,
    n_dimensions: int,
    method: SamplingMethod,
    seed: int,
) -> np.ndarray:
    """
    Tire n_points dans le hypercube unité [0, 1]^{n_dimensions}.

    POURQUOI factoriser cette fonction ?
        IC, BC et résidu ont tous besoin d'un tirage "brut" dans [0, 1]^d
        qu'ils scalent ensuite vers leur domaine physique propre
        (objet, paroi, [0, t*_max]…). Centraliser évite de dupliquer
        le switch sobol/lhs/uniform et garantit le même comportement.

    Parameters
    ----------
    n_points : int
        Nombre de points à tirer.
    n_dimensions : int
        Dimension de l'espace de tirage.
        Exemples :
            2 → (x*, y*) pour l'IC
            2 → (coordonnée_libre, t*_unitaire) pour une paroi BC
            3 → (x*, y*, t*_unitaire) pour le résidu
    method : SamplingMethod
        "sobol" | "lhs" | "uniform".
    seed : int
        Graine de reproductibilité.

    Returns
    -------
    samples : np.ndarray of shape (n_points, n_dimensions)
        Valeurs dans [0, 1].
    """
    if method == "sobol":
        return sobol_sample(n_points, n_dimensions, seed=seed)

    if method == "lhs":
        return latin_hypercube(n_points, n_dimensions, seed=seed)

    if method == "uniform":
        rng = np.random.default_rng(seed)
        return rng.random((n_points, n_dimensions), dtype=np.float64)

    raise ValueError(
        f"Méthode d'échantillonnage inconnue : {method!r}. "
        "Choisir 'sobol', 'lhs' ou 'uniform'."
    )


def _scale_unit_time_to_fourier(
    unit_time: np.ndarray,
    t_star_max: float,
) -> np.ndarray:
    """
    Affine le temps unitaire u ∈ [0, 1] vers le temps de Fourier t*.

        t* = u · t*_max

    POURQUOI cette étape est CRITIQUE (bug fréquent) :
        Si on oublie ce scaling, t* vit dans [0, 1] au lieu de
        [0, t*_max ≈ 0.1]. Le PINN croit alors résoudre l'EDP sur
        un horizon 10× trop long (Fo=1 au lieu de Fo=0.1).

    Parameters
    ----------
    unit_time : np.ndarray
        Tirage dans [0, 1], shape quelconque.
    t_star_max : float
        Borne haute de Fourier (= α t_max / L_ref²).

    Returns
    -------
    t_star : np.ndarray
        Même shape que unit_time, valeurs dans [0, t_star_max].
    """
    return unit_time * t_star_max


def _initial_temperature_field(
    x_star: np.ndarray,
    y_star: np.ndarray,
    cfg: DataConfig,
) -> np.ndarray:
    """
    Construit le champ de température adimensionné à t* = 0.

    Physique :
        On dépose un objet chaud (T = T_obj ⇒ T* = 1) dans une pièce
        initialement à T_amb (T* = 0). La diffusion part de cette
        condition en créneau (disque ou carré).

        T*(x*, y*, 0) = 1_{ (x*,y*) ∈ objet }

    Parameters
    ----------
    x_star, y_star : np.ndarray of shape (n_points,)
        Coordonnées adimensionnées des points IC.
    cfg : DataConfig
        Fournit la géométrie de l'objet (centre, rayon, forme).

    Returns
    -------
    T_star : np.ndarray of shape (n_points,)
        Valeurs dans {0.0, 1.0}.
    """
    # Masque booléen : True = à l'intérieur de l'objet chaud
    inside_hot_object = mask_hot_object(
        x_star,
        y_star,
        center_x=cfg.obj_cx,
        center_y=cfg.obj_cy,
        radius=cfg.obj_radius,
        shape=cfg.obj_shape,
    )

    # Par défaut tout le monde est à T* = 0 (pièce ambiante)…
    T_star = np.zeros(x_star.shape[0], dtype=np.float64)

    # …sauf les points dans l'objet, peints à T* = 1
    T_star[inside_hot_object] = 1.0
    return T_star


def _points_per_wall(n_bc: int, n_walls: int = 4) -> List[int]:
    """
    Répartit n_bc points le plus équitablement possible sur n_walls parois.

    POURQUOI équilibrer ?
        Chaque paroi impose la même condition Dirichlet T*=0. Si une
        paroi reçoit beaucoup moins de points, sa contrainte sera
        sous-représentée dans la loss BC → solution dissymétrique.

    Exemple : n_bc=2000, n_walls=4 → [500, 500, 500, 500]
    Exemple : n_bc=2003, n_walls=4 → [501, 501, 501, 500]
              (le reste 2003 % 4 = 3 est distribué aux 3 premières)

    Parameters
    ----------
    n_bc : int
        Budget total de points BC.
    n_walls : int
        Nombre de parois (4 pour un carré).

    Returns
    -------
    counts : list[int] de longueur n_walls
        counts[i] = nombre de points alloués à la paroi i.
    """
    base_count = n_bc // n_walls
    remainder = n_bc - base_count * n_walls
    return [base_count + (1 if wall_index < remainder else 0) for wall_index in range(n_walls)]


def _sample_one_wall(
    wall_id: int,
    n_points_on_wall: int,
    method: SamplingMethod,
    seed: int,
    t_star_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Échantillonne les points d'UNE paroi du carré unité.

    Sur chaque paroi, une coordonnée est FIXÉE (0 ou 1) et l'autre
    est LIBRE ∈ [0, 1]. Le temps t* est libre dans [0, t*_max].

    Codage des parois :
        0 → gauche  : x* = 0,  y* libre
        1 → droite  : x* = 1,  y* libre
        2 → bas     : y* = 0,  x* libre
        3 → haut    : y* = 1,  x* libre

    Parameters
    ----------
    wall_id : int
        Indice de paroi ∈ {0, 1, 2, 3}.
    n_points_on_wall : int
        Nombre de points à tirer sur cette paroi.
    method : SamplingMethod
        Stratégie quasi-aléatoire.
    seed : int
        Graine (décalée par paroi dans l'appelant pour diversifier).
    t_star_max : float
        Borne haute de Fourier pour scaler t*.

    Returns
    -------
    x_star, y_star, t_star, wall_ids : np.ndarray of shape (n_points_on_wall,)
    """
    # Tirage 2D dans [0, 1]² : (coordonnée libre, temps unitaire)
    free_and_time = _sample_in_unit_hypercube(
        n_points=n_points_on_wall,
        n_dimensions=2,
        method=method,
        seed=seed,
    )
    free_coordinate = free_and_time[:, 0]  # ∈ [0, 1]
    unit_time = free_and_time[:, 1]  # ∈ [0, 1]

    # Scaling du temps vers [0, t*_max]
    t_star = _scale_unit_time_to_fourier(unit_time, t_star_max)

    # Fixation de la coordonnée de paroi
    if wall_id == 0:  # gauche : x* = 0
        x_star = np.zeros(n_points_on_wall, dtype=np.float64)
        y_star = free_coordinate
    elif wall_id == 1:  # droite : x* = 1
        x_star = np.ones(n_points_on_wall, dtype=np.float64)
        y_star = free_coordinate
    elif wall_id == 2:  # bas : y* = 0
        x_star = free_coordinate
        y_star = np.zeros(n_points_on_wall, dtype=np.float64)
    elif wall_id == 3:  # haut : y* = 1
        x_star = free_coordinate
        y_star = np.ones(n_points_on_wall, dtype=np.float64)
    else:
        raise ValueError(f"wall_id invalide : {wall_id}. Attendu dans {{0,1,2,3}}.")

    wall_ids = np.full(n_points_on_wall, wall_id, dtype=np.int64)
    return x_star, y_star, t_star, wall_ids


# ==========================================================================
# API publique : sample_ic / sample_bc / sample_residual / sample_all
# ==========================================================================

def sample_ic(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    T_star_init: Optional[float] = None,
) -> CollocationBatch:
    """
    Échantillonne les points de CONDITION INITIALE (t* = 0).

    Physique
    --------
    À l'instant initial, on connaît le champ de température partout
    dans la pièce :
        - T* = 1  à l'intérieur de l'objet chaud (disque/carré)
        - T* = 0  à l'extérieur (pièce à température ambiante)

    Le réseau devra coller à ces valeurs : L_ic = MSE(T_pred, T*_cible).

    Autograd
    --------
    x*, y*, t* ont requires_grad=True. Même si à l'Étape 1 on ne
    dérive pas encore, c'est déjà le format attendu par l'Étape 2
    (calcul de ∂T*/∂x* etc. si on régularise aussi sur l'IC).

    T*_cible a requires_grad=False : c'est une constante, pas une
    variable du graphe de calcul.

    Parameters
    ----------
    cfg : DataConfig, optional
        Configuration (N_ic, seed, device, géométrie objet…).
        Si None, on utilise DEFAULT_CONFIG.
    method : {'sobol', 'lhs', 'uniform'}
        Stratégie d'échantillonnage spatial de (x*, y*).
    T_star_init : float, optional
        Si fourni, impose une IC UNIFORME (ignore l'objet chaud).
        Utile UNIQUEMENT pour des tests unitaires ciblés.
        En production, laisser None pour activer le masque objet.

    Returns
    -------
    batch : dict[str, torch.Tensor]
        "x_star" : shape (N_ic, 1), requires_grad=True
        "y_star" : shape (N_ic, 1), requires_grad=True
        "t_star" : shape (N_ic, 1), requires_grad=True,  toutes les valeurs = 0
        "T_star" : shape (N_ic, 1), requires_grad=False, valeurs ∈ {0, 1}
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    n_ic = cfg.N_ic

    # --- 1. Tirage spatial (x*, y*) dans le carré unité ----------------
    # Shape : (n_ic, 2)
    xy_unit = _sample_in_unit_hypercube(
        n_points=n_ic,
        n_dimensions=2,
        method=method,
        seed=cfg.seed,
    )
    x_star = xy_unit[:, 0]  # shape (n_ic,)
    y_star = xy_unit[:, 1]  # shape (n_ic,)

    # --- 2. Temps fixé à 0 (définition même de la condition initiale) --
    t_star = np.zeros(n_ic, dtype=np.float64)  # shape (n_ic,)

    # --- 3. Température cible selon le masque objet chaud --------------
    if T_star_init is not None:
        # Mode test : champ uniforme imposé
        T_star = np.full(n_ic, float(T_star_init), dtype=np.float64)
    else:
        # Mode physique : T*=1 dans l'objet, 0 dehors
        T_star = _initial_temperature_field(x_star, y_star, cfg)

    # --- 4. Conversion en tenseurs PyTorch (N, 1) ----------------------
    device = cfg.device
    return {
        "x_star": to_tensor(x_star, requires_grad=True, device=device),
        "y_star": to_tensor(y_star, requires_grad=True, device=device),
        "t_star": to_tensor(t_star, requires_grad=True, device=device),
        "T_star": to_tensor(T_star, requires_grad=False, device=device),
    }


def sample_bc(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    T_star_bc: float = 0.0,
) -> CollocationBatch:
    """
    Échantillonne les points de CONDITIONS AUX LIMITES Dirichlet.

    Physique
    --------
    Les 4 parois de la pièce sont maintenues à T_amb pendant toute la
    simulation ⇒ en adimensionné : T* = 0 sur ∂Ω, pour tout t* ∈ [0, t*_max].

    C'est une condition de Dirichlet homogène. Le réseau devra coller
    à T*_pred = 0 sur ces points : L_bc = MSE(T_pred, 0).

    Répartition
    -----------
    Budget N_bc réparti équitablement sur les 4 parois
    (N_bc // 4 points chacune, reste ventilé sur les premières).

    Temps
    -----
    t* est tiré dans [0, t*_max] (PAS [0, 1]) via le scaling Fourier.

    Parameters
    ----------
    cfg : DataConfig, optional
        Configuration (N_bc, t_star_max, seed, device…).
    method : {'sobol', 'lhs', 'uniform'}
        Stratégie pour (coordonnée libre, temps unitaire).
    T_star_bc : float
        Température adimensionnée imposée au bord (défaut 0.0 = T_amb).

    Returns
    -------
    batch : dict[str, torch.Tensor]
        "x_star" : shape (N_bc, 1), requires_grad=True
        "y_star" : shape (N_bc, 1), requires_grad=True
        "t_star" : shape (N_bc, 1), requires_grad=True,  valeurs ∈ [0, t*_max]
        "T_star" : shape (N_bc, 1), requires_grad=False, valeurs = T_star_bc
        "wall"   : shape (N_bc, 1), dtype=int64,         indices de paroi {0,1,2,3}
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    n_bc = cfg.N_bc
    t_star_max = cfg.t_star_max
    points_per_wall = _points_per_wall(n_bc, n_walls=4)

    # Accumulateurs par paroi (listes de tableaux 1-D)
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    t_list: List[np.ndarray] = []
    wall_list: List[np.ndarray] = []

    for wall_id, n_points_on_wall in enumerate(points_per_wall):
        if n_points_on_wall == 0:
            continue

        # Graine décalée par paroi → sous-échantillons diversifiés
        # mais toujours reproductibles à cfg.seed fixé.
        wall_seed = cfg.seed + wall_id + 1

        x_wall, y_wall, t_wall, wall_ids = _sample_one_wall(
            wall_id=wall_id,
            n_points_on_wall=n_points_on_wall,
            method=method,
            seed=wall_seed,
            t_star_max=t_star_max,
        )
        x_list.append(x_wall)
        y_list.append(y_wall)
        t_list.append(t_wall)
        wall_list.append(wall_ids)

    # Concaténation des 4 parois → shape (n_bc,)
    x_star = np.concatenate(x_list)
    y_star = np.concatenate(y_list)
    t_star = np.concatenate(t_list)
    wall_ids = np.concatenate(wall_list)

    # Cible Dirichlet : T* constant sur tout le bord
    T_star = np.full(n_bc, T_star_bc, dtype=np.float64)

    # Mélange aléatoire pour casser l'ordre paroi-par-paroi.
    # POURQUOI ? Si on laisse l'ordre [gauche…, droite…, bas…, haut…],
    # un mini-batch contigu ne verrait qu'une seule paroi → gradients
    # biaisés. Le shuffle garantit un mélange i.i.d. approximatif.
    rng = np.random.default_rng(cfg.seed)
    permutation = rng.permutation(n_bc)
    x_star = x_star[permutation]
    y_star = y_star[permutation]
    t_star = t_star[permutation]
    wall_ids = wall_ids[permutation]
    T_star = T_star[permutation]

    device = cfg.device
    return {
        "x_star": to_tensor(x_star, requires_grad=True, device=device),
        "y_star": to_tensor(y_star, requires_grad=True, device=device),
        "t_star": to_tensor(t_star, requires_grad=True, device=device),
        "T_star": to_tensor(T_star, requires_grad=False, device=device),
        # wall est un indice entier : pas de grad, dtype int64
        "wall": torch.as_tensor(wall_ids, dtype=torch.int64, device=device).unsqueeze(-1),
    }


def sample_residual(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
    interior_eps: float = 1e-6,
) -> CollocationBatch:
    """
    Échantillonne les points de COLLOCATION pour le résidu PDE.

    Physique
    --------
    À l'intérieur du domaine spatio-temporel, la solution doit vérifier
    l'équation de la chaleur adimensionnée :

        ∂T*/∂t*  −  (∂²T*/∂x*² + ∂²T*/∂y*²)  =  0

    Le résidu r(x*, y*, t*) est exactement le membre de gauche évalué
    sur la prédiction du réseau. La loss résidu est MSE(r, 0).

    Domaine de tirage
    -----------------
        (x*, y*) ∈ (eps, 1 - eps)²          ← strictement intérieur spatial
        t*      ∈ (eps_t, t*_max - eps_t)  ← strictement intérieur temporel

    POURQUOI strictement intérieur ?
        - Éviter le double comptage avec l'IC (t*=0) et les BC (bords).
        - Les dérivées secondes par autograd sont mal définies / bruitées
          pile sur le bord si la condition Dirichlet y est déjà imposée.

    POURQUOI requires_grad=True sur x*, y*, t* ?
        C'est ICI que l'autograd est indispensable. À l'Étape 2 on fera :

            T_pred = network(x*, y*, t*)             # shape (N, 1)
            dT_dt  = grad(T_pred, t*, grad_outputs=ones)[0]   # ∂T*/∂t*
            dT_dx  = grad(T_pred, x*, ...)[0]                  # ∂T*/∂x*
            d2T_dx2 = grad(dT_dx, x*, ...)[0]                  # ∂²T*/∂x*²
            ... idem en y ...
            residual = dT_dt - (d2T_dx2 + d2T_dy2)

        Sans requires_grad=True sur les entrées, grad(...) lève une erreur.

    Parameters
    ----------
    cfg : DataConfig, optional
        Configuration (N_res, t_star_max, seed, device…).
    method : {'sobol', 'lhs', 'uniform'}
        Stratégie 3D.
    interior_eps : float
        Marge relative pour rester strictement intérieur.
        - spatiale : appliquée sur [0, 1] → [eps, 1-eps]
        - temporelle : appliquée sur [0, t*_max] → [eps·t*_max, t*_max·(1-eps)]

    Returns
    -------
    batch : dict[str, torch.Tensor]
        "x_star" : shape (N_res, 1), requires_grad=True, ∈ (0, 1)
        "y_star" : shape (N_res, 1), requires_grad=True, ∈ (0, 1)
        "t_star" : shape (N_res, 1), requires_grad=True, ∈ (0, t*_max)
        (pas de "T_star" : le résidu n'a pas de cible Dirichlet)
    """
    cfg = cfg or DEFAULT_CONFIG
    set_seed(cfg.seed)

    n_res = cfg.N_res
    t_star_max = cfg.t_star_max

    # --- 1. Tirage brut 3D dans [0, 1]³ --------------------------------
    # Shape : (n_res, 3)  colonnes = (x_unit, y_unit, t_unit)
    unit_samples = _sample_in_unit_hypercube(
        n_points=n_res,
        n_dimensions=3,
        method=method,
        seed=cfg.seed + 99,  # décalage pour ne pas corréler avec IC/BC
    )

    # --- 2. Scaling spatial : [0, 1] → (eps, 1 - eps) ------------------
    spatial_lower = interior_eps
    spatial_upper = 1.0 - interior_eps
    x_star = spatial_lower + (spatial_upper - spatial_lower) * unit_samples[:, 0]
    y_star = spatial_lower + (spatial_upper - spatial_lower) * unit_samples[:, 1]

    # --- 3. Scaling temporel : [0, 1] → (eps_t, t*_max - eps_t) --------
    # eps_t proportionnel à t*_max pour rester cohérent si on change
    # t_max ou alpha (la marge relative reste interior_eps).
    time_eps = interior_eps * t_star_max
    time_lower = time_eps
    time_upper = t_star_max - time_eps
    t_star = time_lower + (time_upper - time_lower) * unit_samples[:, 2]

    # --- 4. Conversion en tenseurs avec autograd activé ----------------
    device = cfg.device
    return {
        "x_star": to_tensor(x_star, requires_grad=True, device=device),
        "y_star": to_tensor(y_star, requires_grad=True, device=device),
        "t_star": to_tensor(t_star, requires_grad=True, device=device),
    }


def sample_all(
    cfg: Optional[DataConfig] = None,
    method: SamplingMethod = "sobol",
) -> Dict[str, CollocationBatch]:
    """
    Raccourci : génère IC + BC + résidu d'un seul appel.

    Utile dans un script d'entraînement pour peupler d'un coup
    tout le dataloader PINN.

    Returns
    -------
    bundles : dict
        {
          "ic"  : sortie de sample_ic,
          "bc"  : sortie de sample_bc,
          "res" : sortie de sample_residual,
        }
    """
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
