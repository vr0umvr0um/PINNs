"""
==============================================================================
src/validation.py — Métriques PINN vs solveur de référence (Étape 4)
==============================================================================

Le sujet demande « MSE, L2-relative error ». Deux grandeurs complémentaires :

    MSE     = mean( (T_pinn − T_ref)² )
              erreur ABSOLUE moyenne, en unités de T*² (donc ∈ [0, 1]).
              En °C : racine(MSE) × ΔT.

    L2 rel. = ‖T_pinn − T_ref‖₂ / ‖T_ref‖₂
              erreur RELATIVE, sans dimension : 0.05 = 5 % d'erreur.

POURQUOI LES DEUX ?
-------------------
La température décroît avec le temps (la chaleur diffuse vers les murs
froids) : à t*_max le champ ne vaut plus que quelques pourcents. La MSE
devient alors minuscule MÊME si le PINN se trompe complètement de forme ;
inversement l'erreur relative explose dès que ‖T_ref‖ est petit. On
rapporte donc les deux, ET on les trace en fonction du temps : une erreur
globale unique masquerait que le PINN est bon à t* = 0.01 et mauvais à
t* = 0.1.

ERREUR GLOBALE : la norme est calculée sur TOUS les instants à la fois
(somme des carrés sur l'espace-temps). Elle est donc dominée par les
instants où la température est grande — c'est la norme L²(Ω × [0, t_max])
naturelle d'une EDP parabolique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.fd_solver import FDSolution
from src.inference import predict_over_times
from src.models import PINN

# Garde-fou anti division par zéro (champ de référence identiquement nul).
_EPS = 1e-30


def mse(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Erreur quadratique moyenne (en unités T*²)."""
    return float(np.mean((prediction - reference) ** 2))


def relative_l2_error(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Erreur L2 relative ‖p − r‖₂ / ‖r‖₂ (sans dimension)."""
    numerator = float(np.linalg.norm((prediction - reference).ravel()))
    denominator = float(np.linalg.norm(reference.ravel()))
    return numerator / (denominator + _EPS)


@dataclass
class ValidationReport:
    """
    Bilan complet de la comparaison PINN / référence.

    Attributs
    ---------
    t_star : (n_times,)           instants comparés
    prediction, reference, error : (n_times, n, n)
        error = |prediction − reference|
    mse_per_time, l2_per_time : (n_times,)
    mse_global, l2_global, max_abs_error : float
    negative_fraction : float
        Part des points où le PINN prédit T* < 0 (non physique : le
        principe du maximum interdit T* < 0 pour cette EDP).
    """

    t_star: np.ndarray
    prediction: np.ndarray
    reference: np.ndarray
    error: np.ndarray
    mse_per_time: np.ndarray
    l2_per_time: np.ndarray
    mse_global: float
    l2_global: float
    max_abs_error: float
    negative_fraction: float

    def summary(self, delta_T: float = 1.0) -> str:
        """Résumé multi-lignes ; `delta_T` convertit l'erreur max en °C."""
        lines = [
            f"  MSE globale          : {self.mse_global:.3e}   (RMSE = {np.sqrt(self.mse_global) * delta_T:.3f} °C)",
            f"  Erreur L2 relative   : {self.l2_global:.3e}   ({100 * self.l2_global:.1f} %)",
            f"  Erreur max |ΔT*|     : {self.max_abs_error:.3e}   ({self.max_abs_error * delta_T:.2f} °C)",
            f"  Points T* < 0        : {100 * self.negative_fraction:.1f} %",
        ]
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, object]:
        """Version JSON-sérialisable."""
        return {
            "mse_global": self.mse_global,
            "l2_global": self.l2_global,
            "max_abs_error": self.max_abs_error,
            "negative_fraction": self.negative_fraction,
            "t_star": self.t_star.tolist(),
            "mse_per_time": self.mse_per_time.tolist(),
            "l2_per_time": self.l2_per_time.tolist(),
        }


def compare_fields(
    t_star: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> ValidationReport:
    """Calcule toutes les métriques à partir de deux piles (n_times, n, n)."""
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Shapes incompatibles : prédiction {prediction.shape} "
            f"vs référence {reference.shape}."
        )
    error = np.abs(prediction - reference)
    n_times = prediction.shape[0]
    mse_per_time = np.array([mse(prediction[k], reference[k]) for k in range(n_times)])
    l2_per_time = np.array(
        [relative_l2_error(prediction[k], reference[k]) for k in range(n_times)]
    )
    return ValidationReport(
        t_star=np.asarray(t_star, dtype=np.float64),
        prediction=prediction,
        reference=reference,
        error=error,
        mse_per_time=mse_per_time,
        l2_per_time=l2_per_time,
        mse_global=mse(prediction, reference),
        l2_global=relative_l2_error(prediction, reference),
        max_abs_error=float(error.max()),
        negative_fraction=float(np.mean(prediction < 0.0)),
    )


def validate_pinn(model: PINN, reference: FDSolution) -> ValidationReport:
    """
    Évalue le PINN sur les NŒUDS de la grille de référence, aux instants
    de la référence, puis calcule les métriques.

    Aucune interpolation : PINN et différences finies sont comparés aux
    mêmes points (x_i, y_j, t_k).
    """
    prediction = predict_over_times(model, reference.x_star, reference.y_star, reference.t_star)
    return compare_fields(reference.t_star, prediction, reference.T_star)


def metrics_at_times(report: ValidationReport, t_values: List[float]) -> List[Dict[str, float]]:
    """Extrait MSE / L2 aux instants enregistrés les plus proches de t_values."""
    rows = []
    for t_value in t_values:
        k = int(np.argmin(np.abs(report.t_star - t_value)))
        rows.append(
            {
                "t_star": float(report.t_star[k]),
                "mse": float(report.mse_per_time[k]),
                "l2_rel": float(report.l2_per_time[k]),
            }
        )
    return rows


def plot_validation(
    report: ValidationReport,
    path,
    *,
    title: str = "Étape 4 — PINN vs différences finies",
    times_to_show=(0.0, 0.01, 0.05, 0.1),
):
    """
    Figure de validation : cartes PINN / référence / erreur à quelques
    instants, puis erreurs MSE et L2 relative en fonction du temps.

    Les deux premières colonnes partagent la MÊME échelle de couleur (celle
    de la référence) : sinon un PINN qui prédit un champ presque nul
    paraîtrait « bon » grâce à une autoscale trompeuse.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from pathlib import Path

    t_max = float(report.t_star[-1])
    indices = []
    for t_value in times_to_show:
        if t_value <= t_max + 1e-12:
            index = int(np.argmin(np.abs(report.t_star - t_value)))
            if index not in indices:
                indices.append(index)
    n_rows = len(indices)

    figure = plt.figure(figsize=(13, 3.0 * n_rows + 3.6), constrained_layout=True)
    grid = figure.add_gridspec(n_rows + 1, 3, height_ratios=[1.0] * n_rows + [1.15])
    extent = (0.0, 1.0, 0.0, 1.0)

    for row, index in enumerate(indices):
        reference = report.reference[index]
        vmax = max(float(reference.max()), 1e-6)
        panels = (
            (report.prediction[index], "PINN", dict(cmap="inferno", vmin=0.0, vmax=vmax)),
            (reference, "Différences finies", dict(cmap="inferno", vmin=0.0, vmax=vmax)),
            (report.error[index], "|erreur|", dict(cmap="viridis", vmin=0.0)),
        )
        for column, (field, label, style) in enumerate(panels):
            axis = figure.add_subplot(grid[row, column])
            # .T : imshow place le 1er indice en lignes ; on veut x en abscisse.
            image = axis.imshow(field.T, origin="lower", extent=extent, **style)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            axis.set_title(f"{label} — t* = {report.t_star[index]:.3f}", fontsize=10)
            axis.set_xlabel("x*")
            axis.set_ylabel("y*")

    axis_l2 = figure.add_subplot(grid[n_rows, 0:2])
    axis_l2.plot(report.t_star, report.l2_per_time, "o-", color="tab:red", ms=3)
    axis_l2.axhline(report.l2_global, color="gray", ls="--", lw=1,
                    label=f"globale = {report.l2_global:.3f}")
    axis_l2.set_xlabel("t*")
    axis_l2.set_ylabel("erreur L2 relative")
    axis_l2.set_title("Erreur L2 relative en fonction du temps")
    axis_l2.legend()
    axis_l2.grid(alpha=0.3)

    axis_mse = figure.add_subplot(grid[n_rows, 2])
    axis_mse.semilogy(report.t_star, np.maximum(report.mse_per_time, 1e-16), "o-",
                      color="tab:blue", ms=3)
    axis_mse.set_xlabel("t*")
    axis_mse.set_ylabel("MSE (T*²)")
    axis_mse.set_title(f"MSE globale = {report.mse_global:.2e}")
    axis_mse.grid(alpha=0.3, which="both")

    figure.suptitle(title, fontsize=13)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=130)
    plt.close(figure)
    return target.resolve()


__all__ = [
    "plot_validation",
    "ValidationReport",
    "compare_fields",
    "metrics_at_times",
    "mse",
    "relative_l2_error",
    "validate_pinn",
]
