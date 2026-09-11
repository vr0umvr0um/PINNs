"""
Configuration centrale du projet PINN — diffusion thermique 2D instationnaire.

Toutes les grandeurs physiques, géométriques et d'échantillonnage sont
regroupées dans la dataclass `DataConfig`. L'adimensionnement transforme
l'équation de la chaleur :

    ∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)

en sa forme normalisée (Fourier) :

    ∂T*/∂t* = ∂²T*/∂x*² + ∂²T*/∂y*²

avec :
    x* = x / Lx,  y* = y / Ly,
    t* = α t / L_ref²,  L_ref = max(Lx, Ly),
    T* = (T - T_amb) / (T_obj - T_amb).

Domaine adimensionné : (x*, y*) ∈ [0, 1]², t* ∈ [0, t*_max], T* ∈ [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DataConfig:
    """Constantes physiques, géométriques et paramètres d'échantillonnage."""

    # --- Géométrie physique (m) ---
    Lx: float = 1.0
    Ly: float = 1.0

    # --- Physique ---
    alpha: float = 2e-5  # diffusivité thermique (m²/s), ex. air ~2e-5
    T_amb: float = 20.0  # température ambiante / parois (°C)
    T_obj: float = 80.0  # température de référence objet chaud (°C)

    # --- Temps physique de simulation (s) ---
    t_max: float = 5000.0  # horizon temporel physique

    # --- Budgets d'échantillonnage ---
    N_ic: int = 2000  # points condition initiale (t* = 0)
    N_bc: int = 2000  # points conditions aux limites (4 parois)
    N_res: int = 20000  # points de collocation (résidu PDE)

    # --- Domaine adimensionné ---
    x_star_range: Tuple[float, float] = (0.0, 1.0)
    y_star_range: Tuple[float, float] = (0.0, 1.0)
    t_star_range: Tuple[float, float] = (0.0, 1.0)  # t* normalisé sur [0, 1]

    # --- Reproductibilité ---
    seed: int = 42

    # --- Device PyTorch ---
    device: str = "cpu"

    # Métadonnées (non utilisées pour le sampling, utiles au logging)
    project_name: str = field(default="pinn-thermal-2d")

    # ------------------------------------------------------------------
    # Propriétés dérivées — adimensionnement
    # ------------------------------------------------------------------
    @property
    def L_ref(self) -> float:
        """Longueur de référence pour le nombre de Fourier."""
        return max(self.Lx, self.Ly)

    @property
    def delta_T(self) -> float:
        """Écart de température de référence (°C)."""
        return self.T_obj - self.T_amb

    @property
    def t_ref(self) -> float:
        """Temps de référence (s) : t_ref = L_ref² / α."""
        return (self.L_ref**2) / self.alpha

    @property
    def t_star_max(self) -> float:
        """
        Temps adimensionné maximal correspondant à t_max physique.

        t* = α t / L_ref² = t / t_ref.
        On normalise ensuite sur [0, 1] via t*_norm = t* / t*_max
        pour que le réseau travaille sur un cube unitaire ; le facteur
        t*_max est réinjecté dans le résidu si besoin (Étapes ultérieures).
        Ici t_star_range reste [0, 1] par convention d'échantillonnage.
        """
        return self.t_max / self.t_ref

    def to_x_star(self, x: float) -> float:
        """x → x* = x / Lx."""
        return x / self.Lx

    def to_y_star(self, y: float) -> float:
        """y → y* = y / Ly."""
        return y / self.Ly

    def to_t_star(self, t: float) -> float:
        """
        t (s) → t*_norm ∈ [0, 1].

        t*_phys = α t / L_ref², puis t*_norm = t*_phys / t*_max = t / t_max.
        """
        return t / self.t_max

    def to_T_star(self, T: float) -> float:
        """T (°C) → T* = (T - T_amb) / (T_obj - T_amb)."""
        return (T - self.T_amb) / self.delta_T

    def from_T_star(self, T_star: float) -> float:
        """T* → T (°C)."""
        return T_star * self.delta_T + self.T_amb

    def summary(self) -> str:
        """Résumé lisible de la configuration."""
        lines = [
            "=" * 60,
            f"  {self.project_name} — DataConfig",
            "=" * 60,
            f"  Géométrie     : Lx={self.Lx} m, Ly={self.Ly} m",
            f"  Diffusivité   : α={self.alpha:.2e} m²/s",
            f"  Températures  : T_amb={self.T_amb}°C, T_obj={self.T_obj}°C",
            f"  ΔT            : {self.delta_T}°C",
            f"  t_max         : {self.t_max} s  (t*_max physique = {self.t_star_max:.4f})",
            f"  t_ref = L²/α  : {self.t_ref:.1f} s",
            f"  Échantillonnage: N_ic={self.N_ic}, N_bc={self.N_bc}, N_res={self.N_res}",
            f"  Domaine *     : x*{self.x_star_range}, y*{self.y_star_range}, "
            f"t*{self.t_star_range}",
            f"  Seed / device : {self.seed} / {self.device}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Instance par défaut, importable partout
DEFAULT_CONFIG = DataConfig()
