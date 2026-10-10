"""
==============================================================================
src/inference.py — Recharger un PINN entraîné et l'évaluer sur une grille
==============================================================================

POURQUOI CE MODULE ?
--------------------
L'entraînement (Étape 3) écrit des checkpoints ; la validation (Étape 4) et
le démonstrateur (Étape 5) doivent les RELIRE. Recharger un `state_dict`
exige de reconstruire d'abord un réseau de MÊME forme. Trois sources, par
ordre de priorité :

    1. `extra["architecture"]`  — écrite par PINNTrainer (checkpoints récents) ;
    2. déduction depuis les shapes du state_dict — pour les anciens
       checkpoints (largeur, profondeur, bornes de normalisation) ; seule
       l'ACTIVATION est invisible dans un state_dict, d'où le paramètre
       `fallback_activation` ;
    3. échec explicite sinon.

On récupère aussi la `DataConfig` d'entraînement (`extra["data_cfg"]`) : elle
décrit l'objet chaud sur lequel le réseau a appris. Sans elle, on ne peut
pas savoir si l'objet demandé dans l'interface est celui du modèle.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from config import DataConfig
from src.models import PINN
from src.utils import load_checkpoint


def data_config_from_dict(description: Optional[Dict[str, Any]]) -> Optional[DataConfig]:
    """
    Reconstruit une DataConfig depuis un dict (champs inconnus ignorés).

    POURQUOI filtrer ? Un checkpoint écrit par une version future (ou
    passée) du projet peut contenir des champs que cette version ne connaît
    pas : on préfère ignorer ces champs plutôt que de planter au chargement.
    """
    if not description:
        return None
    known = {field.name for field in fields(DataConfig)}
    kwargs = {key: value for key, value in description.items() if key in known}
    for key in ("x_star_range", "y_star_range"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return DataConfig(**kwargs)


def _infer_architecture(
    state_dict: Dict[str, torch.Tensor],
    fallback_activation: str,
    fallback_t_star_max: float,
) -> Dict[str, Any]:
    """Déduit l'architecture depuis les shapes d'un state_dict de PINN."""
    linear_keys = sorted(
        (key for key in state_dict if key.startswith("network.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    if len(linear_keys) < 2:
        raise ValueError(
            "State_dict non reconnu : impossible de déduire l'architecture "
            "(attendu : clés 'network.<i>.weight')."
        )
    first_weight = state_dict[linear_keys[0]]  # (H, 3)
    description: Dict[str, Any] = {
        "n_hidden_layers": len(linear_keys) - 1,
        "n_neurons": int(first_weight.shape[0]),
        "activation": fallback_activation,
        "normalize_inputs": "normalizer.lower" in state_dict,
    }
    if description["normalize_inputs"]:
        lower = state_dict["normalizer.lower"].flatten()
        span = state_dict["normalizer.span"].flatten()
        description["input_lower"] = tuple(float(v) for v in lower)
        description["input_upper"] = tuple(float(v) for v in (lower + span))
    else:
        description["input_upper"] = (1.0, 1.0, fallback_t_star_max)
    return description


def load_trained_model(
    path: Union[str, Path],
    *,
    fallback_activation: str = "tanh",
    device: Union[str, torch.device] = "cpu",
) -> Tuple[PINN, Optional[DataConfig], Dict[str, Any]]:
    """
    Recharge un PINN depuis un checkpoint `best_model.pt` / `last_model.pt`.

    Parameters
    ----------
    path : str or Path
        Checkpoint produit par `PINNTrainer`.
    fallback_activation : str
        Activation à supposer si le checkpoint ne la mentionne pas
        (anciens checkpoints). Ignoré si `extra["architecture"]` existe.
    device : str or torch.device
        Device cible du modèle retourné (mode `eval`, gradients des
        paramètres conservés pour permettre le calcul du résidu).

    Returns
    -------
    model : PINN
    data_cfg : DataConfig or None
        Configuration physique d'entraînement, si elle a été sauvegardée.
    payload : dict
        Le checkpoint brut (epoch, best_loss, history, extra…).
    """
    payload = load_checkpoint(path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    extra = payload.get("extra") or {}

    data_cfg = data_config_from_dict(extra.get("data_cfg"))
    architecture = extra.get("architecture")
    if architecture is None:
        t_star_max = data_cfg.t_star_max if data_cfg is not None else 0.1
        architecture = _infer_architecture(state_dict, fallback_activation, t_star_max)

    model = PINN.from_architecture(architecture)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, data_cfg, payload


def predict_grid(
    model: PINN,
    x_star: np.ndarray,
    y_star: np.ndarray,
    t_star: float,
) -> np.ndarray:
    """
    Évalue T*_θ sur la grille produit cartésien x_star × y_star à t* fixé.

    Returns
    -------
    np.ndarray of shape (len(x_star), len(y_star)) — indexing "ij",
    même convention que `FDSolution.T_star[k]`.
    """
    device = next(model.parameters()).device
    grid_x, grid_y = np.meshgrid(x_star, y_star, indexing="ij")
    flat_x = torch.as_tensor(grid_x.reshape(-1, 1), dtype=torch.float32, device=device)
    flat_y = torch.as_tensor(grid_y.reshape(-1, 1), dtype=torch.float32, device=device)
    flat_t = torch.full_like(flat_x, float(t_star))
    values = model.predict(flat_x, flat_y, flat_t)
    return values.cpu().numpy().reshape(grid_x.shape).astype(np.float64)


def predict_over_times(
    model: PINN,
    x_star: np.ndarray,
    y_star: np.ndarray,
    t_star: np.ndarray,
) -> np.ndarray:
    """Pile de `predict_grid` pour plusieurs instants → (n_times, n_x, n_y)."""
    return np.stack([predict_grid(model, x_star, y_star, float(t)) for t in t_star])


__all__ = [
    "data_config_from_dict",
    "load_trained_model",
    "predict_grid",
    "predict_over_times",
]
