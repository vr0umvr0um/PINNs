"""
==============================================================================
src/ — Package source du projet PINN thermal 2D
==============================================================================

Organisation pédagogique des modules
------------------------------------
    utils.py      Échantillonnage quasi-aléatoire (Sobol, LHS) et masques
                  géométriques (intérieur, bord, objet chaud). Couche
                  "mathématique pure" : aucun savoir sur IC/BC/résidu.

    sampling.py   Assemblage des points de collocation PINN :
                  sample_ic, sample_bc, sample_residual.
                  S'appuie sur utils + config.DataConfig.

À l'Étape 2 (à venir) on ajoutera typiquement :
    models.py     Architecture du réseau (MLP).
    physics.py    Résidu PDE via autograd (∂T*/∂t* − Δ*T*).
    losses.py     L_ic + L_bc + L_res pondérées.
"""

from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import (
    latin_hypercube,
    mask_boundary,
    mask_hot_object,
    mask_interior,
    sobol_sample,
)

__all__ = [
    "sample_ic",
    "sample_bc",
    "sample_residual",
    "sobol_sample",
    "latin_hypercube",
    "mask_boundary",
    "mask_hot_object",
    "mask_interior",
]

__version__ = "0.1.0"
