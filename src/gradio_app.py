"""
==============================================================================
src/gradio_app.py — Interface Gradio du démonstrateur (Étape 5)
==============================================================================

Ce module ne contient QUE le câblage des widgets ; la logique (géométrie,
référence DF, métriques, rendu, ré-entraînement) est dans `src/demo.py`.

Ce que le sujet demande, et où le trouver dans l'interface :

    « faire varier les paramètres de l'objet »      → forme, centre, rayon
    « carte de chaleur prédite avec un slider t »   → slider de temps (secondes)
    « afficher les cartes d'erreur »                → 3e carte + coupe + erreur L2(t)

Les paramètres physiques (α, T_amb, T_obj) agissent instantanément : ils
sont absorbés par l'adimensionnement et ne demandent aucun ré-entraînement.
La géométrie, elle, est dans la condition initiale : le bouton
« Ré-entraîner » adapte le réseau au nouvel objet.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gradio as gr
import matplotlib

# Backend non interactif : Gradio sert les requêtes depuis des threads, où un
# backend GUI (macOS) planterait. Les figures sont de toute façon des images.
matplotlib.use("Agg")

from config import DataConfig
from src.demo import (
    RETRAIN_PRESETS,
    clone_model,
    compare_for_object,
    describe_object,
    metrics_text,
    object_config,
    physical_time_horizon,
    render_snapshot,
    retrain_for_object,
    seconds_to_t_star,
)
from src.models import PINN

INTRO = """
# 🌡️ PINN thermique 2D — démonstrateur

Un objet chaud est posé dans une pièce de 1 m × 1 m dont les murs restent à
température ambiante. Un **réseau de neurones informé par la physique (PINN)**
prédit la diffusion de la chaleur ; on le compare à un **solveur classique aux
différences finies**.

* **α, T_amb, T_obj** : effet immédiat (le réseau résout l'équation adimensionnée).
* **Forme / position / rayon de l'objet** : le réseau est entraîné pour *un* objet.
  Pour un autre objet, la référence est recalculée instantanément mais le PINN
  doit être **ré-entraîné** (bouton en bas à gauche).
"""


def build_app(
    model: PINN,
    trained_cfg: DataConfig,
    *,
    n_grid: int = 81,
    n_times: int = 41,
    source: str = "",
) -> gr.Blocks:
    """
    Construit l'application Gradio.

    Parameters
    ----------
    model : PINN
        Réseau pré-entraîné (état de départ de chaque session).
    trained_cfg : DataConfig
        Configuration physique de son entraînement (objet chaud, α, t*_max…).
    n_grid, n_times : int
        Résolution de la grille de comparaison et nombre d'instants.
    source : str
        Provenance affichée (chemin du checkpoint).
    """
    base_cfg = trained_cfg

    def refresh(
        shape: str, center_x: float, center_y: float, radius: float,
        alpha: float, T_amb: float, T_obj: float, seconds: float,
        state_model: PINN, state_cfg: DataConfig,
    ) -> Tuple[Any, str]:
        """Recalcule figure et métriques pour l'ensemble des widgets."""
        try:
            object_cfg = object_config(base_cfg, shape, center_x, center_y, radius)
        except ValueError as error:
            raise gr.Error(str(error))
        if T_obj <= T_amb:
            raise gr.Error("T_obj doit être strictement supérieur à T_amb.")
        report = compare_for_object(state_model, object_cfg, n_grid=n_grid, n_times=n_times)
        t_star = seconds_to_t_star(base_cfg, alpha, seconds)
        figure = render_snapshot(
            report, object_cfg, t_star, alpha=alpha, T_amb=T_amb, T_obj=T_obj,
        )
        text = metrics_text(report, t_star, object_cfg, state_cfg, T_amb=T_amb, T_obj=T_obj)
        return figure, text

    def on_alpha(alpha: float, seconds: float) -> Any:
        """Adapte la plage du slider de temps à la diffusivité choisie."""
        if alpha is None or alpha <= 0:
            raise gr.Error("α doit être > 0.")
        horizon = physical_time_horizon(base_cfg, alpha)
        return gr.update(maximum=horizon, value=min(seconds, horizon), step=horizon / 200)

    def on_retrain(
        shape: str, center_x: float, center_y: float, radius: float,
        preset: str, progress: gr.Progress = gr.Progress(),
    ):
        try:
            object_cfg = object_config(base_cfg, shape, center_x, center_y, radius)
        except ValueError as error:
            raise gr.Error(str(error))
        result = retrain_for_object(
            object_cfg, preset,
            progress=lambda fraction, message: progress(fraction, desc=message),
        )
        losses = result.final_losses
        status = (
            f"✅ Réseau ré-entraîné pour **{describe_object(result.data_cfg)}** en "
            f"{result.elapsed_seconds:.0f} s — perte finale {losses['loss_total']:.2e} "
            f"(IC {losses['loss_ic']:.1e}, BC {losses['loss_bc']:.1e}, "
            f"résidu {losses['loss_res']:.1e})."
        )
        return result.model, result.data_cfg, status

    def on_reset() -> Tuple[PINN, DataConfig, str]:
        return clone_model(model), trained_cfg, f"↩️ Réseau pré-entraîné restauré ({describe_object(trained_cfg)})."

    with gr.Blocks(title="PINN thermique 2D") as app:
        gr.Markdown(INTRO)

        # Un callable est évalué à l'ouverture de CHAQUE session : chacune
        # reçoit sa propre copie du réseau (un ré-entraînement n'affecte
        # pas les autres visiteurs). Un nn.Module nu serait pris pour un
        # callable de chargement, d'où le lambda.
        state_model = gr.State(lambda: clone_model(model))
        state_cfg = gr.State(lambda: trained_cfg)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Objet chaud")
                shape = gr.Radio(
                    [("Disque", "disk"), ("Carré", "square")],
                    value=trained_cfg.obj_shape, label="Forme",
                )
                center_x = gr.Slider(0.1, 0.9, value=trained_cfg.obj_cx, step=0.01, label="Centre x/L")
                center_y = gr.Slider(0.1, 0.9, value=trained_cfg.obj_cy, step=0.01, label="Centre y/L")
                radius = gr.Slider(
                    0.05, 0.4, value=trained_cfg.obj_radius, step=0.01,
                    label="Rayon (disque) / demi-côté (carré), en fraction de L",
                )

                gr.Markdown("### Physique (effet immédiat)")
                alpha = gr.Number(value=trained_cfg.alpha, label="Diffusivité α (m²/s) — air ≈ 2e-5")
                T_amb = gr.Slider(-20, 40, value=trained_cfg.T_amb, step=1, label="T ambiante (°C)")
                T_obj = gr.Slider(30, 200, value=trained_cfg.T_obj, step=1, label="T objet (°C)")

                gr.Markdown("### Réseau")
                preset = gr.Dropdown(list(RETRAIN_PRESETS), value=list(RETRAIN_PRESETS)[0],
                                     label="Budget de ré-entraînement")
                retrain_button = gr.Button("🔁 Ré-entraîner le PINN pour cet objet", variant="primary")
                reset_button = gr.Button("↩️ Restaurer le réseau pré-entraîné")
                status = gr.Markdown(
                    f"Réseau pré-entraîné : **{describe_object(trained_cfg)}**"
                    + (f"  \n`{source}`" if source else "")
                )

            with gr.Column(scale=3):
                horizon = physical_time_horizon(base_cfg, trained_cfg.alpha)
                seconds = gr.Slider(
                    0.0, horizon, value=horizon * 0.2, step=horizon / 200,
                    label="Temps t (secondes)",
                )
                plot = gr.Plot(label="Température")
                metrics = gr.Markdown()

        inputs = [shape, center_x, center_y, radius, alpha, T_amb, T_obj, seconds, state_model, state_cfg]
        outputs = [plot, metrics]

        # Déplacer le slider de temps : rendu immédiat (référence en cache).
        seconds.change(refresh, inputs, outputs, show_progress="hidden", trigger_mode="always_last")
        # Paramètres d'objet : on attend le relâchement du slider (.release)
        # pour ne pas recalculer la référence à chaque pixel.
        for widget in (center_x, center_y, radius):
            widget.release(refresh, inputs, outputs, show_progress="hidden")
        for widget in (shape, T_amb, T_obj):
            widget.change(refresh, inputs, outputs, show_progress="hidden")
        alpha.change(on_alpha, [alpha, seconds], [seconds]).then(refresh, inputs, outputs, show_progress="hidden")

        retrain_button.click(
            on_retrain, [shape, center_x, center_y, radius, preset], [state_model, state_cfg, status],
        ).then(refresh, inputs, outputs)
        reset_button.click(on_reset, None, [state_model, state_cfg, status]).then(refresh, inputs, outputs)

        app.load(refresh, inputs, outputs)

    return app


__all__ = ["build_app", "INTRO"]
