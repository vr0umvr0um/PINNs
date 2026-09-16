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

    models.py     Architecture du réseau T_θ(x*, y*, t*) : MLP à
                  activations C² (obligatoire pour dériver deux fois)
                  et normalisation affine des entrées vers [-1, 1].

    physics.py    Dérivation automatique (torch.autograd) et résidu de
                  l'EDP : r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²).
                  Fournit aussi une solution analytique de référence
                  pour VALIDER l'autograd.

    losses.py     Loss multi-objectif L = w_ic·L_ic + w_bc·L_bc + w_res·L_res.

Chaîne de dépendances (aucun cycle) :
    utils → sampling
    models → physics → losses

À l'Étape 3 (à venir) on ajoutera typiquement :
    training.py   Boucle hybride Adam → L-BFGS, checkpoints, pondération
                  dynamique et Residual Adaptive Resampling.
"""

from src.losses import (
    LossTerms,
    loss_boundary,
    loss_initial,
    loss_residual,
    pinn_loss,
)
from src.models import PINN, build_activation
from src.physics import (
    analytic_solution,
    gradient,
    heat_derivatives,
    pde_residual,
    relative_residual_error,
)
from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import (
    latin_hypercube,
    mask_boundary,
    mask_hot_object,
    mask_interior,
    sobol_sample,
)

__all__ = [
    # Étape 1 — échantillonnage
    "sample_ic",
    "sample_bc",
    "sample_residual",
    "sobol_sample",
    "latin_hypercube",
    "mask_boundary",
    "mask_hot_object",
    "mask_interior",
    # Étape 2 — architecture, autograd, loss
    "PINN",
    "build_activation",
    "gradient",
    "heat_derivatives",
    "pde_residual",
    "analytic_solution",
    "relative_residual_error",
    "LossTerms",
    "loss_initial",
    "loss_boundary",
    "loss_residual",
    "pinn_loss",
]

__version__ = "0.2.0"
