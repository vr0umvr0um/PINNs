"""
==============================================================================
src/ — Package source du projet PINN thermal 2D
==============================================================================

Organisation pédagogique des modules
------------------------------------
    utils.py      Boîte à outils transverse : échantillonnage quasi-aléatoire
                  (Sobol, LHS), masques géométriques, sélection du device
                  (get_device), checkpoints (save/load) et figures
                  d'apprentissage (plot_loss_history).

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

    trainer.py    Boucle d'entraînement hybride Adam → L-BFGS (Étape 3) :
                  classe PINNTrainer, early stopping, pondération dynamique
                  (fixed / grad_norm / lr_annealing), suivi TensorBoard/tqdm,
                  checkpoints best/last + history.json.

    fd_solver.py  Solveur de référence aux différences finies (Étape 4),
                  schéma explicite FTCS avec condition CFL.

    inference.py  Rechargement d'un PINN depuis un checkpoint (architecture
                  et DataConfig embarquées) et évaluation sur grille.

    validation.py Métriques PINN vs référence : MSE, erreur L2 relative,
                  par instant et globales ; figure de validation.

    demo.py       Logique du démonstrateur (Étape 5), testable sans
                  navigateur : géométrie, rendu, ré-entraînement.

    gradio_app.py Câblage de l'interface Gradio (importé à la demande :
                  `import src` ne requiert pas Gradio).

Chaîne de dépendances (aucun cycle) :
    utils → sampling
    models → physics → losses → trainer
    fd_solver, inference → validation → demo → gradio_app
"""

from src.fd_solver import FDSolution, solve_heat_fd
from src.inference import load_trained_model, predict_grid
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
from src.trainer import HISTORY_KEYS, FitResult, PINNTrainer
from src.validation import ValidationReport, relative_l2_error, validate_pinn
from src.utils import (
    format_seconds,
    get_device,
    latin_hypercube,
    load_checkpoint,
    mask_boundary,
    mask_hot_object,
    mask_interior,
    plot_loss_history,
    save_checkpoint,
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
    # Étape 3 — entraînement, suivi, checkpoints, figures
    "PINNTrainer",
    "FitResult",
    "HISTORY_KEYS",
    "get_device",
    "format_seconds",
    "save_checkpoint",
    "load_checkpoint",
    "plot_loss_history",
    # Étape 4 — validation numérique
    "FDSolution",
    "solve_heat_fd",
    "load_trained_model",
    "predict_grid",
    "ValidationReport",
    "validate_pinn",
    "relative_l2_error",
]

__version__ = "0.5.0"
