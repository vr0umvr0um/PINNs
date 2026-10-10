"""
==============================================================================
src/demo.py — Logique du démonstrateur interactif (Étape 5), sans Gradio
==============================================================================

POURQUOI SÉPARER LA LOGIQUE DE L'INTERFACE ?
---------------------------------------------
`src/gradio_app.py` ne fait que brancher des widgets sur les fonctions de
ce module. Toute la substance — géométrie, référence DF, métriques, rendu
des cartes, ré-entraînement — vit ici, donc se TESTE sans navigateur.

CE QUE L'UTILISATEUR PEUT FAIRE VARIER, ET CE QUE CELA COÛTE
-------------------------------------------------------------
Le PINN résout l'équation ADIMENSIONNÉE. Trois paramètres « physiques »
sont donc GRATUITS (simple post-traitement, aucun ré-entraînement) :

    α  (diffusivité)        t = t* · L² / α        → seule l'échelle de temps change
    T_amb, T_obj            T = T_amb + T*·(T_obj − T_amb)   → échelle de couleur

C'est la conséquence directe de l'adimensionnement de l'Étape 1 : un seul
réseau sert tous les matériaux et toutes les températures.

La GÉOMÉTRIE de l'objet (forme, centre, rayon), elle, est inscrite dans la
condition initiale, donc dans la loss : un réseau entraîné pour un disque
centré ne connaît RIEN d'un carré décentré. Deux réponses honnêtes :

    1. on évalue quand même le réseau existant et on MONTRE l'erreur
       (le désaccord avec la référence DF est la preuve visible que le
       réseau n'est valable que pour son objet d'entraînement) ;
    2. on propose un RÉ-ENTRAÎNEMENT court (`retrain_for_object`) pour le
       nouvel objet, à budget réduit (quelques minutes sur CPU).

La référence DF, elle, est recalculée pour tout objet en ~0.2 s.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from matplotlib.figure import Figure

from config import DataConfig, LossConfig, ModelConfig, TrainConfig
from src.fd_solver import FDSolution, solve_heat_fd
from src.inference import predict_over_times
from src.models import PINN
from src.sampling import sample_all
from src.trainer import PINNTrainer
from src.validation import ValidationReport, compare_fields

# Marge minimale entre l'objet et les parois : un objet qui touche un mur
# rendrait la condition initiale incompatible avec T* = 0 sur le bord.
WALL_MARGIN: float = 0.02

# Poids de la loss pour le ré-entraînement : IC et BC surpondérées (10/10/1).
# Mesuré sur l'objet par défaut : L2 globale 34 % avec 1/1/1, 25 % avec 10/10/1
# (cf. README, Étape 3). L'IC en créneau est le terme le plus raide.
DEMO_LOSS_CONFIG = LossConfig(w_ic=10.0, w_bc=10.0, w_res=1.0)

# Budgets de ré-entraînement proposés dans l'interface.
# (epochs Adam, itérations L-BFGS, N_ic, N_bc, N_res)
RETRAIN_PRESETS: Dict[str, Tuple[int, int, int, int, int]] = {
    "Rapide (~1 min)": (800, 20, 1000, 1000, 5000),
    "Standard (~13 min)": (3000, 100, 2000, 2000, 20000),
}


# ==========================================================================
# Géométrie
# ==========================================================================

def object_config(
    base: DataConfig,
    shape: str,
    center_x: float,
    center_y: float,
    radius: float,
) -> DataConfig:
    """
    Copie de `base` avec un autre objet chaud, après validation.

    Raises
    ------
    ValueError
        Forme inconnue, rayon ≤ 0, ou objet qui sort de la pièce / touche
        une paroi (à WALL_MARGIN près).
    """
    if shape not in ("disk", "square"):
        raise ValueError(f"Forme inconnue : {shape!r} (attendu 'disk' ou 'square').")
    if radius <= 0.0:
        raise ValueError(f"Le rayon doit être > 0, reçu {radius}.")
    low, high = WALL_MARGIN, 1.0 - WALL_MARGIN
    if not (
        center_x - radius >= low and center_x + radius <= high
        and center_y - radius >= low and center_y + radius <= high
    ):
        raise ValueError(
            "L'objet doit tenir dans la pièce, à au moins "
            f"{WALL_MARGIN} d'une paroi (centre=({center_x:.2f}, {center_y:.2f}), "
            f"rayon={radius:.2f})."
        )
    return replace(
        base,
        obj_shape=shape,
        obj_cx=float(center_x),
        obj_cy=float(center_y),
        obj_radius=float(radius),
    )


def same_object(first: DataConfig, second: DataConfig, tolerance: float = 1e-6) -> bool:
    """Vrai si les deux configs décrivent le même objet chaud."""
    return (
        first.obj_shape == second.obj_shape
        and abs(first.obj_cx - second.obj_cx) <= tolerance
        and abs(first.obj_cy - second.obj_cy) <= tolerance
        and abs(first.obj_radius - second.obj_radius) <= tolerance
    )


def describe_object(cfg: DataConfig) -> str:
    """Libellé court : « disque, centre (0.50, 0.50), r = 0.15 »."""
    name = "disque" if cfg.obj_shape == "disk" else "carré"
    size = "r" if cfg.obj_shape == "disk" else "demi-côté"
    return f"{name}, centre ({cfg.obj_cx:.2f}, {cfg.obj_cy:.2f}), {size} = {cfg.obj_radius:.2f}"


# ==========================================================================
# Comparaison PINN / référence (avec cache de la référence)
# ==========================================================================

_REFERENCE_CACHE: "OrderedDict[tuple, FDSolution]" = OrderedDict()
_REFERENCE_CACHE_SIZE = 16
_REFERENCE_LOCK = threading.Lock()


def reference_solution(cfg: DataConfig, n_grid: int, n_times: int) -> FDSolution:
    """
    Solution DF pour l'objet de `cfg`, mémoïsée (LRU).

    POURQUOI UN CACHE ? Déplacer le slider de temps ne change pas l'objet :
    inutile de recalculer 5 000 pas de temps. Seule une modification de
    la géométrie invalide l'entrée.
    """
    key = (cfg.obj_shape, round(cfg.obj_cx, 6), round(cfg.obj_cy, 6),
           round(cfg.obj_radius, 6), round(cfg.t_star_max, 9), n_grid, n_times)
    with _REFERENCE_LOCK:
        cached = _REFERENCE_CACHE.get(key)
        if cached is not None:
            _REFERENCE_CACHE.move_to_end(key)
            return cached
    solution = solve_heat_fd(cfg, n_grid=n_grid, n_times=n_times)
    with _REFERENCE_LOCK:
        _REFERENCE_CACHE[key] = solution
        while len(_REFERENCE_CACHE) > _REFERENCE_CACHE_SIZE:
            _REFERENCE_CACHE.popitem(last=False)
    return solution


def compare_for_object(
    model: PINN,
    object_cfg: DataConfig,
    n_grid: int = 81,
    n_times: int = 41,
) -> ValidationReport:
    """
    Évalue `model` et la référence DF pour l'objet `object_cfg`, à tous
    les instants. Le slider de temps n'a ensuite qu'à indexer le résultat.
    """
    reference = reference_solution(object_cfg, n_grid, n_times)
    prediction = predict_over_times(model, reference.x_star, reference.y_star, reference.t_star)
    return compare_fields(reference.t_star, prediction, reference.T_star)


# ==========================================================================
# Conversions « gratuites » (post-traitement des grandeurs adimensionnées)
# ==========================================================================

def physical_time_horizon(cfg: DataConfig, alpha: float) -> float:
    """
    Horizon physique (s) couvert par le domaine d'entraînement t* ∈ [0, t*_max]
    pour une diffusivité `alpha` : t_max = t*_max · L² / α.
    """
    if alpha <= 0.0:
        raise ValueError(f"alpha doit être > 0, reçu {alpha}.")
    return cfg.t_star_max * cfg.L_ref**2 / alpha


def seconds_to_t_star(cfg: DataConfig, alpha: float, seconds: float) -> float:
    """t* = α t / L², borné au domaine d'entraînement [0, t*_max]."""
    t_star = alpha * seconds / cfg.L_ref**2
    return float(min(max(t_star, 0.0), cfg.t_star_max))


def to_celsius(T_star: np.ndarray, T_amb: float, T_obj: float) -> np.ndarray:
    """T = T_amb + T*·(T_obj − T_amb)."""
    return T_amb + T_star * (T_obj - T_amb)


# ==========================================================================
# Rendu
# ==========================================================================

def render_snapshot(
    report: ValidationReport,
    object_cfg: DataConfig,
    t_star: float,
    *,
    alpha: float,
    T_amb: float,
    T_obj: float,
) -> Figure:
    """
    Figure du démonstrateur à l'instant t* :
        ligne 1 : carte PINN | carte référence DF | carte d'erreur absolue
        ligne 2 : coupe horizontale à y = centre de l'objet | erreur L2(t)

    Les cartes PINN et DF partagent la même échelle de couleur (celle de
    la référence) : une autoscale indépendante ferait paraître « correct »
    un PINN qui se trompe d'amplitude.
    """
    index = int(np.argmin(np.abs(report.t_star - t_star)))
    delta_T = T_obj - T_amb
    seconds = report.t_star[index] * object_cfg.L_ref**2 / alpha

    prediction = to_celsius(report.prediction[index], T_amb, T_obj)
    reference = to_celsius(report.reference[index], T_amb, T_obj)
    error = report.error[index] * abs(delta_T)

    n_grid = reference.shape[0]
    nodes = np.linspace(0.0, 1.0, n_grid)
    vmin, vmax = float(min(T_amb, T_obj)), float(max(reference.max(), T_amb + 1e-9))

    figure = Figure(figsize=(13, 7.2), layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=[1.15, 1.0])
    extent = (0.0, 1.0, 0.0, 1.0)

    def draw_map(column: int, field: np.ndarray, title: str, cmap: str, **limits) -> None:
        axis = figure.add_subplot(grid[0, column])
        # .T : le 1er indice du champ est x → il doit aller en abscisse.
        image = axis.imshow(field.T, origin="lower", extent=extent, cmap=cmap, **limits)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02, label="°C")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x/L")
        axis.set_ylabel("y/L")

    draw_map(0, prediction, "PINN", "inferno", vmin=vmin, vmax=vmax)
    draw_map(1, reference, "Référence (différences finies)", "inferno", vmin=vmin, vmax=vmax)
    draw_map(2, error, "|PINN − référence|", "viridis", vmin=0.0)

    axis_cut = figure.add_subplot(grid[1, 0:2])
    row = int(np.argmin(np.abs(nodes - object_cfg.obj_cy)))
    axis_cut.plot(nodes, reference[:, row], "-", color="black", lw=2, label="Référence DF")
    axis_cut.plot(nodes, prediction[:, row], "--", color="tab:red", lw=2, label="PINN")
    axis_cut.set_xlabel("x/L")
    axis_cut.set_ylabel("T (°C)")
    axis_cut.set_title(f"Coupe à y/L = {nodes[row]:.2f}", fontsize=11)
    axis_cut.legend(loc="upper right")
    axis_cut.grid(alpha=0.3)

    axis_error = figure.add_subplot(grid[1, 2])
    axis_error.plot(report.t_star, 100.0 * report.l2_per_time, color="tab:red")
    axis_error.axvline(report.t_star[index], color="gray", ls="--")
    axis_error.set_xlabel("t*")
    axis_error.set_ylabel("erreur L2 relative (%)")
    axis_error.set_title("Erreur L2 relative au cours du temps", fontsize=11)
    axis_error.grid(alpha=0.3)

    figure.suptitle(
        f"t = {format_duration(seconds)}   (t* = {report.t_star[index]:.4f})"
        f"   —   {describe_object(object_cfg)}",
        fontsize=12,
    )
    return figure


def format_duration(seconds: float) -> str:
    """Durée lisible : « 42 s », « 12 min 05 s », « 1 h 23 min »."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes} min {rest:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def metrics_text(
    report: ValidationReport,
    t_star: float,
    object_cfg: DataConfig,
    trained_cfg: DataConfig,
    *,
    T_amb: float,
    T_obj: float,
) -> str:
    """Bloc Markdown de métriques ; avertit si l'objet ≠ objet d'entraînement."""
    index = int(np.argmin(np.abs(report.t_star - t_star)))
    delta_T = abs(T_obj - T_amb)
    lines = [
        f"**À cet instant (t\\* = {report.t_star[index]:.4f})** : "
        f"MSE = {report.mse_per_time[index]:.2e} T\\*², "
        f"erreur L2 relative = {100 * report.l2_per_time[index]:.1f} %, "
        f"erreur max = {report.error[index].max() * delta_T:.2f} °C",
        "",
        f"**Sur tout l'horizon** : MSE = {report.mse_global:.2e} T\\*² "
        f"(RMSE = {np.sqrt(report.mse_global) * delta_T:.2f} °C), "
        f"erreur L2 relative = {100 * report.l2_global:.1f} %, "
        f"températures non physiques (T\\* < 0) : {100 * report.negative_fraction:.1f} % des points",
    ]
    if not same_object(object_cfg, trained_cfg):
        lines += [
            "",
            f"⚠️ **Ce réseau a été entraîné pour : {describe_object(trained_cfg)}.** "
            "Il est évalué ici sur un autre objet : l'erreur élevée est normale. "
            "Cliquer sur « Ré-entraîner » pour l'adapter.",
        ]
    return "\n".join(lines)


# ==========================================================================
# Ré-entraînement
# ==========================================================================

@dataclass
class RetrainResult:
    """Modèle ré-entraîné et configuration physique associée."""

    model: PINN
    data_cfg: DataConfig
    elapsed_seconds: float
    final_losses: Dict[str, float]


def retrain_for_object(
    object_cfg: DataConfig,
    preset: str = "Rapide (~1 min)",
    *,
    model_cfg: Optional[ModelConfig] = None,
    loss_cfg: Optional[LossConfig] = None,
    weighting: str = "fixed",
    progress: Optional[Callable[[float, str], None]] = None,
    checkpoint_dir: Optional[str] = None,
) -> RetrainResult:
    """
    Entraîne un PINN neuf pour l'objet de `object_cfg` (CPU, full-batch).

    `progress(fraction, message)` est appelé ~2 fois par seconde depuis le
    thread APPELANT pendant que l'entraînement tourne dans un thread de
    fond : l'interface reste réactive et peut afficher une barre.
    """
    if preset not in RETRAIN_PRESETS:
        raise ValueError(f"Préréglage inconnu : {preset!r}. Choix : {list(RETRAIN_PRESETS)}.")
    adam_epochs, lbfgs_iterations, n_ic, n_bc, n_res = RETRAIN_PRESETS[preset]

    cfg = replace(object_cfg, N_ic=n_ic, N_bc=n_bc, N_res=n_res, device="cpu")
    model_cfg = model_cfg or ModelConfig()
    batches = sample_all(cfg)
    model = PINN.from_config(cfg, model_cfg)
    train_cfg = TrainConfig(
        adam_epochs=adam_epochs,
        lbfgs_iterations=lbfgs_iterations,
        weighting=weighting,
        early_stopping=False,
    )
    import tempfile

    work_dir = checkpoint_dir or tempfile.mkdtemp(prefix="pinn_demo_")
    trainer = PINNTrainer(
        model, batches["ic"], batches["bc"], batches["res"],
        loss_cfg or LossConfig(), train_cfg,
        device="cpu", checkpoint_dir=work_dir, use_tqdm=False, verbose=False,
        metadata={"data_cfg": _to_plain_dict(cfg)},
    )

    outcome: Dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = trainer.fit()
        except BaseException as error:  # remonté dans le thread appelant
            outcome["error"] = error

    start = time.perf_counter()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    total = adam_epochs + lbfgs_iterations
    while thread.is_alive():
        thread.join(timeout=0.5)
        if progress is not None:
            done = len(trainer.history["loss_total"])
            last = trainer.history["loss_total"][-1] if done else float("nan")
            phase = "Adam" if done <= adam_epochs else "L-BFGS"
            progress(min(done / max(total, 1), 1.0), f"{phase} — itération {done}/{total} — perte {last:.3e}")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]

    trainer.restore_best_weights()
    model.eval()
    return RetrainResult(
        model=model,
        data_cfg=cfg,
        elapsed_seconds=time.perf_counter() - start,
        final_losses=trainer.evaluate(),
    )


def _to_plain_dict(cfg: DataConfig) -> Dict[str, object]:
    from dataclasses import asdict

    return asdict(cfg)


def clone_model(model: PINN) -> PINN:
    """Copie profonde (l'état Gradio ne doit pas partager le modèle global)."""
    return copy.deepcopy(model)


__all__ = [
    "DEMO_LOSS_CONFIG",
    "RETRAIN_PRESETS",
    "RetrainResult",
    "WALL_MARGIN",
    "clone_model",
    "compare_for_object",
    "describe_object",
    "format_duration",
    "metrics_text",
    "object_config",
    "physical_time_horizon",
    "reference_solution",
    "render_snapshot",
    "retrain_for_object",
    "same_object",
    "seconds_to_t_star",
    "to_celsius",
]
