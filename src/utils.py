"""
==============================================================================
src/utils.py — Échantillonnage quasi-aléatoire & masques géométriques
==============================================================================

POURQUOI CE MODULE
------------------
Un PINN évalue son résidu PDE sur un nuage de points de collocation.
La QUALITÉ de ce nuage conditionne la qualité de l'apprentissage :

    - Tirage uniforme naïf  → grumeaux, trous (mauvaise couverture).
    - Sobol / LHS           → faible discrépance, couverture homogène.

On fournit aussi des MASQUES géométriques (intérieur, bord, objet chaud)
qui traduisent en booléens NumPy les régions physiques du problème.

Convention de shapes dans tout le projet
----------------------------------------
    points Sobol/LHS : np.ndarray de shape (n_points, n_dimensions)
    masques          : np.ndarray de shape (n_points,)  dtype=bool
    tenseurs PINN    : torch.Tensor de shape (n_points, 1)
                       ↑ la dimension "1" est le canal scalaire
                         (une seule coordonnée par tenseur : x* OU y* OU t*)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.stats import qmc


# ==========================================================================
# Reproductibilité
# ==========================================================================

def set_seed(seed: int = 42) -> None:
    """
    Fixe les graines NumPy et PyTorch pour rendre les tirages reproductibles.

    POURQUOI c'est indispensable en PINN / ML :
        Sans graine fixe, chaque run tire des points différents → la loss
        et les figures changent, et on ne peut plus comparer deux expériences
        ni debugger un comportement bizarre.

    Parameters
    ----------
    seed : int
        Entier quelconque (42 par convention communautaire).
        La même graine ⇒ les mêmes séquences pseudo-aléatoires.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Sur GPU, il y a une graine séparée par device
        torch.cuda.manual_seed_all(seed)


# ==========================================================================
# Échantillonnage quasi-aléatoire
# ==========================================================================

def sobol_sample(
    n_points: int,
    n_dimensions: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
    scramble: bool = True,
) -> np.ndarray:
    """
    Tire `n_points` dans [0, 1]^{n_dimensions} via une séquence de Sobol.

    POURQUOI Sobol plutôt qu'un tirage uniforme ?
        Une séquence de Sobol est à FAIBLE DISCRÉPANCE : elle remplit
        l'hypercube plus régulièrement qu'un tirage i.i.d. uniforme.
        Pour un budget de points limité (surtout N_res), cela réduit
        la variance du résidu PDE estimé et accélère la convergence.

    POURQUOI le "scramble" Owen ?
        Le scramble randomise la séquence tout en gardant sa faible
        discrépance. On obtient ainsi plusieurs réalisations indépendantes
        (grâce à `seed`) sans perdre les bonnes propriétés de couverture.

    POURQUOI tirer une puissance de 2 puis tronquer ?
        Les séquences de Sobol sont théoriquement équilibrées sur des
        blocs de taille 2^k. SciPy le recommande ; on tire donc le
        prochain 2^k ≥ n_points, puis on garde les n_points premiers.

    Parameters
    ----------
    n_points : int
        Nombre de points souhaités.
    n_dimensions : int
        Dimension de l'espace (2 pour (x*,y*), 3 pour (x*,y*,t*), …).
    bounds : np.ndarray of shape (n_dimensions, 2), optional
        Bornes (min, max) par dimension. Si None, on reste dans [0, 1]^d.
        Exemple pour scaler le temps vers [0, t*_max] :
            bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, t_star_max]])
    seed : int
        Graine du scramble.
    scramble : bool
        Active le scramble Owen (recommandé : True).

    Returns
    -------
    samples : np.ndarray of shape (n_points, n_dimensions)
        Coordonnées float64 dans le pavé demandé.
    """
    if n_points <= 0:
        return np.zeros((0, n_dimensions), dtype=np.float64)

    # Moteur quasi-Monte-Carlo Sobol (SciPy)
    sobol_engine = qmc.Sobol(
        d=n_dimensions,
        scramble=scramble,
        seed=seed,
    )

    # Prochaine puissance de 2 ≥ n_points (équilibre de la séquence)
    # Ex. n_points=2000 → n_power_of_two = 2048 = 2^11
    n_power_of_two = 1 << int(np.ceil(np.log2(max(n_points, 1))))

    # Tirage puis troncature aux n_points demandés
    # Shape : (n_power_of_two, n_dimensions) → (n_points, n_dimensions)
    unit_samples = sobol_engine.random(n_power_of_two)[:n_points]

    # Scaling optionnel : [0, 1]^d → pavé défini par bounds
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        expected_shape = (n_dimensions, 2)
        if bounds.shape != expected_shape:
            raise ValueError(
                f"`bounds` doit avoir la shape {expected_shape}, "
                f"reçu {bounds.shape}."
            )
        # qmc.scale fait : out = lo + (hi - lo) * unit
        lower_bounds = bounds[:, 0]  # shape (n_dimensions,)
        upper_bounds = bounds[:, 1]  # shape (n_dimensions,)
        unit_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return unit_samples.astype(np.float64)


def latin_hypercube(
    n_points: int,
    n_dimensions: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS) dans [0, 1]^{n_dimensions}.

    POURQUOI le LHS ?
        Le LHS garantit que, sur CHAQUE axe, les n_points se répartissent
        en n_points strates équiprobables (une par « ligne » / « colonne »).
        Moins régulier que Sobol en haute dimension, mais excellent en 2D/3D
        et très utilisé en plan d'expériences.

    Parameters
    ----------
    n_points : int
        Nombre de points.
    n_dimensions : int
        Dimension de l'espace.
    bounds : np.ndarray of shape (n_dimensions, 2), optional
        Bornes (min, max) par dimension. None → [0, 1]^d.
    seed : int
        Graine de générateur.

    Returns
    -------
    samples : np.ndarray of shape (n_points, n_dimensions)
    """
    if n_points <= 0:
        return np.zeros((0, n_dimensions), dtype=np.float64)

    lhs_engine = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    unit_samples = lhs_engine.random(n_points)  # shape (n_points, n_dimensions)

    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        expected_shape = (n_dimensions, 2)
        if bounds.shape != expected_shape:
            raise ValueError(
                f"`bounds` doit avoir la shape {expected_shape}, "
                f"reçu {bounds.shape}."
            )
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        unit_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return unit_samples.astype(np.float64)


# ==========================================================================
# Masques géométriques
# ==========================================================================
# Un "masque" est un tableau de booléens de même longueur que les points.
# True  = le point appartient à la région
# False = le point est hors de la région
# On s'en sert pour :
#   - peindre T*=1 dans l'objet chaud (condition initiale)
#   - vérifier qu'un point BC est bien sur une paroi
#   - filtrer un sous-ensemble pour un plot ou une loss partielle

def mask_interior(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Masque des points STRICTEMENT intérieurs au carré unité [0, 1]².

    POURQUOI "strictement" (ouvert) ?
        Les points de bord sont réservés aux conditions aux limites (BC).
        Le résidu PDE se calcule dans l'OUVERT (0, 1)² pour éviter de
        compter deux fois la même contrainte (BC + résidu sur le même point).

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées, même shape (n_points,) ou broadcastable.
    tolerance : float
        Marge numérique : on exige d'être à plus de `tolerance` du bord.
        Évite qu'un point à 1e-16 du bord soit classé intérieur par erreur
        d'arrondi flottant.

    Returns
    -------
    mask : np.ndarray of bool, same shape as x_star
        True  ↔ point strictement intérieur.
    """
    inside_x = (x_star > tolerance) & (x_star < 1.0 - tolerance)
    inside_y = (y_star > tolerance) & (y_star < 1.0 - tolerance)
    return inside_x & inside_y


def mask_boundary(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Masque des points situés sur le BORD du carré unité [0, 1]².

    Un point est « au bord » si AU MOINS une coordonnée vaut 0 ou 1
    (à `tolerance` près), tout en restant dans le pavé.

    POURQUOI une tolérance plus large (1e-6) que pour l'intérieur ?
        Les points BC sont construits EXACTEMENT à 0 ou 1, mais après
        conversion float32 (PyTorch) la valeur peut dériver légèrement
        (ex. 0.99999994). 1e-6 absorbe cette erreur sans attraper
        d'intérieur légitime.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées.
    tolerance : float
        Tolérance au bord.

    Returns
    -------
    mask : np.ndarray of bool
        True ↔ point sur une paroi (ou un coin).
    """
    on_vertical_wall = (
        (np.abs(x_star - 0.0) <= tolerance) | (np.abs(x_star - 1.0) <= tolerance)
    )
    on_horizontal_wall = (
        (np.abs(y_star - 0.0) <= tolerance) | (np.abs(y_star - 1.0) <= tolerance)
    )
    inside_bounding_box = (
        (x_star >= -tolerance)
        & (x_star <= 1.0 + tolerance)
        & (y_star >= -tolerance)
        & (y_star <= 1.0 + tolerance)
    )
    return inside_bounding_box & (on_vertical_wall | on_horizontal_wall)


def mask_hot_object(
    x_star: np.ndarray,
    y_star: np.ndarray,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius: float = 0.15,
    shape: str = "disk",
) -> np.ndarray:
    """
    Masque des points situés À L'INTÉRIEUR de l'objet chaud.

    POURQUOI ce masque est central pour l'Étape 1 :
        La condition initiale du problème physique est :
            T*(x*, y*, t*=0) = 1   si (x*, y*) ∈ objet chaud
            T*(x*, y*, t*=0) = 0   sinon
        Ce masque est EXACTEMENT la traduction numérique de cette
        discontinuité. sample_ic() s'en sert pour peindre T*.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées des points à tester.
        Shape typique : (n_points,).
    center_x, center_y : float
        Centre de l'objet en coordonnées adimensionnées (x*, y*).
    radius : float
        - si shape="disk"   : rayon du disque
        - si shape="square" : demi-côté du carré (norme infinie)
    shape : {'disk', 'square'}
        Géométrie de l'objet.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_points,)
        True  → point dans l'objet → on imposera T* = 1
        False → point hors objet  → on imposera T* = 0
    """
    # Vecteur centre → point, composante par composante
    delta_x = np.asarray(x_star, dtype=np.float64) - center_x
    delta_y = np.asarray(y_star, dtype=np.float64) - center_y

    if shape == "disk":
        # Disque : ||(dx, dy)||_2 ≤ radius
        distance_squared = delta_x * delta_x + delta_y * delta_y
        return distance_squared <= radius**2

    if shape == "square":
        # Carré aligné sur les axes : ||(dx, dy)||_∞ ≤ radius
        return (np.abs(delta_x) <= radius) & (np.abs(delta_y) <= radius)

    raise ValueError(
        f"Forme d'objet inconnue : {shape!r}. Choisir 'disk' ou 'square'."
    )


def which_boundary(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Identifie LA paroi d'appartenance de chaque point.

    Codage entier (utile pour colorer un plot ou pondérer une loss) :
        0 = paroi gauche   (x* = 0)
        1 = paroi droite   (x* = 1)
        2 = paroi bas      (y* = 0)
        3 = paroi haut     (y* = 1)
       -1 = pas un point de bord

    POURQUOI une priorité x > y sur les coins ?
        Un coin appartient à DEUX parois. Pour éviter un double comptage
        dans les histos / losses, on le rattache arbitrairement à la
        paroi verticale (x). Le choix est conventionnel, l'important
        est qu'il soit déterministe.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées.
    tolerance : float
        Tolérance au bord.

    Returns
    -------
    wall_labels : np.ndarray of int64, same shape as x_star
    """
    # Initialise tout à "non-bord"
    wall_labels = np.full(np.broadcast(x_star, y_star).shape, -1, dtype=np.int64)

    # Parois verticales en premier (priorité coins)
    wall_labels = np.where(np.abs(x_star - 0.0) <= tolerance, 0, wall_labels)
    wall_labels = np.where(np.abs(x_star - 1.0) <= tolerance, 1, wall_labels)

    # Parois horizontales seulement si pas déjà tagué vertical
    not_on_vertical = wall_labels < 0
    wall_labels = np.where(
        not_on_vertical & (np.abs(y_star - 0.0) <= tolerance), 2, wall_labels
    )
    wall_labels = np.where(
        not_on_vertical & (np.abs(y_star - 1.0) <= tolerance), 3, wall_labels
    )
    return wall_labels


# ==========================================================================
# Conversion NumPy → PyTorch
# ==========================================================================

def to_tensor(
    array: np.ndarray,
    requires_grad: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convertit un tableau NumPy en tenseur PyTorch colonne.

    POURQUOI shape (N, 1) et pas (N,) ?
        PyTorch et les MLP attendent en général un tenseur 2D
        (batch_size, n_features). Ici chaque coordonnée (x*, y* ou t*)
        est une "feature" scalaire → n_features = 1.
        Shape finale : (n_points, 1).

    POURQUOI requires_grad=True sur les COORDONNÉES ?
        C'est LE point clé d'un PINN. Pour calculer le résidu
            r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)
        on a besoin des dérivées de la sortie du réseau par rapport
        à ses ENTRÉES (x*, y*, t*). Autograd ne les construira que si
        ces entrées ont requires_grad=True.

        En revanche la CIBLE T* (IC/BC) a requires_grad=False : c'est
        une constante physique, pas une variable d'optimisation.

    Parameters
    ----------
    array : np.ndarray
        Données source, shape (N,) ou (N, 1) ou (N, C).
    requires_grad : bool
        Active le suivi autograd (True pour x*, y*, t* ; False pour T*).
    device : str
        "cpu" ou "cuda".
    dtype : torch.dtype
        float32 par défaut (bon compromis précision / vitesse / VRAM).

    Returns
    -------
    tensor : torch.Tensor
        Shape (N, 1) si l'entrée était 1-D, sinon shape d'origine.
        dtype et device conformes aux arguments.
    """
    tensor = torch.as_tensor(array, dtype=dtype, device=device)

    # Si on reçoit un vecteur 1-D (N,), on l'amène en colonne (N, 1)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)  # (N,) → (N, 1)

    # Active (ou non) le graphe de calcul autograd sur ce tenseur
    tensor = tensor.requires_grad_(requires_grad)
    return tensor


def unit_cube_bounds(n_dimensions: int) -> np.ndarray:
    """
    Construit le tableau de bornes [[0, 1], [0, 1], …] pour n_dimensions.

    Utile pour appeler sobol_sample / latin_hypercube avec un pavé unité
    explicite (plutôt que bounds=None), ce qui rend l'intention lisible.

    Parameters
    ----------
    n_dimensions : int
        Nombre de dimensions.

    Returns
    -------
    bounds : np.ndarray of shape (n_dimensions, 2)
        Chaque ligne vaut [0.0, 1.0].
    """
    return np.tile(np.array([[0.0, 1.0]]), (n_dimensions, 1))


__all__ = [
    "set_seed",
    "sobol_sample",
    "latin_hypercube",
    "mask_interior",
    "mask_boundary",
    "mask_hot_object",
    "which_boundary",
    "to_tensor",
    "unit_cube_bounds",
]
