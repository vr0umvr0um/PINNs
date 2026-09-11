"""
Package source du projet PINN thermal 2D.

Modules :
    utils     — échantillonnage quasi-aléatoire (Sobol / LHS) et masques géométriques
    sampling  — génération des points IC, BC et résidu PDE
"""

from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import (
    latin_hypercube,
    mask_boundary,
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
    "mask_interior",
]

__version__ = "0.1.0"
