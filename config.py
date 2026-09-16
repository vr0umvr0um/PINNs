"""
==============================================================================
config.py — Configuration physique et adimensionnement du PINN thermique 2D
==============================================================================

POURQUOI CE FICHIER EXISTE
--------------------------
Un PINN (Physics-Informed Neural Network) apprend la solution d'une EDP en
minimisant un résidu. Pour que l'apprentissage soit numériquement stable, on
travaille en GRANDEURS ADIMENSIONNÉES (ordre de grandeur ~ 1), pas en SI brutes.

Équation physique de départ (diffusion de la chaleur) :

    ∂T/∂t = α (∂²T/∂x² + ∂²T/∂y²)          [SI : °C, m, s]

Après adimensionnement (variables étoilées *) :

    ∂T*/∂t* = ∂²T*/∂x*² + ∂²T*/∂y*²         [sans dimension]

Le coefficient α disparaît : il est absorbé dans la définition de t*.
C'est le grand intérêt pédagogique et numérique de l'adimensionnement.

Définitions (à retenir pour la soutenance) :

    x* = x / Lx
    y* = y / Ly
    t* = α t / L_ref²          ← nombre de Fourier Fo
    T* = (T - T_amb) / (T_obj - T_amb)

Domaine adimensionné :
    (x*, y*) ∈ [0, 1]²
    t*      ∈ [0, t*_max]      avec t*_max = α t_max / L_ref² ≈ 0.1
    T*      ∈ [0, 1]

Condition initiale physique :
    un objet chaud (T* = 1) est plongé dans une pièce à T_amb (T* = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

# Forme géométrique de l'objet chaud à t* = 0.
# "disk"   → disque de rayon obj_radius centré en (obj_cx, obj_cy)
# "square" → carré de demi-côté obj_radius centré idem
ObjectShape = Literal["disk", "square"]


@dataclass
class DataConfig:
    """
    Conteneur UNIQUE de toutes les constantes du projet.

    POURQUOI une dataclass plutôt que des constantes globales ?
        1. Un seul objet à passer aux fonctions (pas de pollution du namespace).
        2. Facile à surcharger en tests (ex. N_ic=200 pour aller vite).
        3. Les grandeurs DÉRIVÉES (t_ref, t_star_max, delta_T) sont calculées
           à la volée via @property → jamais de risque d'incohérence.

    Convention de nommage :
        - grandeurs physiques SI  : Lx, alpha, T_amb, t_max…
        - grandeurs adimensionnées : x_star, t_star_max, T_star…
        - budgets d'échantillonnage : N_ic, N_bc, N_res
    """

    # ------------------------------------------------------------------
    # Géométrie de la pièce (mètres)
    # ------------------------------------------------------------------
    # Pièce carrée 1 m × 1 m → après adim., le domaine spatial est le
    # carré unité [0, 1]², ce qui simplifie énormément le code.
    Lx: float = 1.0  # largeur de la pièce (m)
    Ly: float = 1.0  # profondeur de la pièce (m)

    # ------------------------------------------------------------------
    # Physique thermique
    # ------------------------------------------------------------------
    # α (alpha) = diffusivité thermique = k / (ρ cp)
    # Ordre de grandeur de l'air stagnant : ~2×10⁻⁵ m²/s.
    # C'est α qui fixe l'échelle de temps de la diffusion
    # (temps caractéristique t_ref = L² / α).
    alpha: float = 2e-5  # m²/s

    # Températures de référence pour l'adimensionnement de T.
    # T_amb : température des parois (Dirichlet) et de l'air initial.
    # T_obj : température de l'objet chaud déposé à t = 0.
    T_amb: float = 20.0  # °C — ambiant / parois
    T_obj: float = 80.0  # °C — objet chaud

    # ------------------------------------------------------------------
    # Horizon temporel physique
    # ------------------------------------------------------------------
    # t_max = 5000 s ≈ 1 h 23 min.
    # Avec α = 2e-5 et L = 1 m :
    #   t*_max = α t_max / L² = 0.1
    # → on n'observe qu'une fraction du temps de diffusion complet (Fo=1).
    t_max: float = 5000.0  # secondes

    # ------------------------------------------------------------------
    # Objet chaud (condition initiale) — coordonnées ADIMENSIONNÉES
    # ------------------------------------------------------------------
    # POURQUOI en coordonnées * ?
    #   Parce que tout le sampling travaille déjà en (x*, y*). Placer
    #   l'objet directement en * évite des conversions aller-retour.
    obj_shape: ObjectShape = "disk"
    obj_cx: float = 0.5  # centre x* de l'objet (milieu de la pièce)
    obj_cy: float = 0.5  # centre y*
    obj_radius: float = 0.15  # rayon (disk) ou demi-côté (square)

    # ------------------------------------------------------------------
    # Budgets d'échantillonnage PINN
    # ------------------------------------------------------------------
    # Un PINN a BESOIN de trois familles de points :
    #   N_ic  : coller à la condition initiale      (t* = 0)
    #   N_bc  : coller aux conditions aux limites   (parois, T* = 0)
    #   N_res : coller à l'EDP dans le volume        (résidu PDE ≈ 0)
    #
    # En pratique N_res >> N_ic, N_bc car le résidu est la contrainte
    # la plus "difficile" (dérivées secondes via autograd).
    N_ic: int = 2000  # points de condition initiale
    N_bc: int = 2000  # points de conditions aux limites (4 parois)
    N_res: int = 20000  # points de collocation (résidu PDE)

    # ------------------------------------------------------------------
    # Domaine spatial adimensionné (toujours le carré unité ici)
    # ------------------------------------------------------------------
    x_star_range: Tuple[float, float] = (0.0, 1.0)
    y_star_range: Tuple[float, float] = (0.0, 1.0)

    # ------------------------------------------------------------------
    # Reproductibilité et matériel
    # ------------------------------------------------------------------
    seed: int = 42  # graine NumPy / PyTorch / Sobol
    device: str = "cpu"  # "cpu" ou "cuda" si GPU disponible

    # Nom du projet (logging, titres de figures)
    project_name: str = field(default="pinn-thermal-2d")

    # ==================================================================
    # Propriétés dérivées — adimensionnement
    # (calculées à la volée, jamais stockées → pas d'incohérence possible)
    # ==================================================================

    @property
    def L_ref(self) -> float:
        """
        Longueur de référence pour le nombre de Fourier.

        POURQUOI max(Lx, Ly) ?
            On veut une seule échelle de longueur. Pour une pièce non
            carrée, prendre le plus grand côté garantit que x*, y* ≤ 1.
        """
        return max(self.Lx, self.Ly)

    @property
    def delta_T(self) -> float:
        """
        Écart de température de référence (°C).

        Sert de dénominateur pour T* :
            T* = (T - T_amb) / delta_T
        Ainsi T_amb → T*=0 et T_obj → T*=1.
        """
        return self.T_obj - self.T_amb

    @property
    def t_ref(self) -> float:
        """
        Temps de référence (s) : t_ref = L_ref² / α.

        Interprétation physique :
            c'est le temps caractéristique pour que la chaleur diffuse
            sur une distance L_ref. Pour L=1 m, α=2e-5 m²/s :
                t_ref = 1 / 2e-5 = 50 000 s ≈ 14 heures.
        """
        return (self.L_ref**2) / self.alpha

    @property
    def t_star_max(self) -> float:
        """
        Temps adimensionné maximal = nombre de Fourier à t = t_max.

            t*_max = α · t_max / L_ref² = t_max / t_ref

        Exemple numérique (config par défaut) :
            t_max = 5000 s, t_ref = 50 000 s  →  t*_max = 0.1

        POINT CRITIQUE POUR LA SOUTENANCE :
            L'échantillonnage de t* (BC et résidu) DOIT vivre dans
            [0, t*_max] = [0, 0.1], et NON dans [0, 1].
            Confondre les deux revient à simuler 10× trop longtemps.
        """
        return self.t_max / self.t_ref

    @property
    def t_star_range(self) -> Tuple[float, float]:
        """
        Intervalle temporel adimensionné effectif : [0, t*_max].

        Exposé comme propriété pour être cohérent avec x_star_range /
        y_star_range, et pour que le résumé / les plots affichent la
        bonne borne haute automatiquement.
        """
        return (0.0, self.t_star_max)

    # ==================================================================
    # Conversions SI ↔ adimensionné
    # ==================================================================

    def to_x_star(self, x: float) -> float:
        """
        Convertit une abscisse physique en adimensionnée.

            x* = x / Lx

        Parameters
        ----------
        x : float
            Abscisse en mètres, typiquement dans [0, Lx].

        Returns
        -------
        float
            x* ∈ [0, 1] si x ∈ [0, Lx].
        """
        return x / self.Lx

    def to_y_star(self, y: float) -> float:
        """
        Convertit une ordonnée physique en adimensionnée.

            y* = y / Ly
        """
        return y / self.Ly

    def to_t_star(self, t: float) -> float:
        """
        Convertit un temps physique (s) en temps de Fourier t*.

            t* = α t / L_ref² = t / t_ref

        POURQUOI diviser par t_ref et pas par t_max ?
            Parce que t* DOIT être le vrai nombre de Fourier pour que
            l'équation adimensionnée n'ait PAS de coefficient devant
            le laplacien. Si on normalisait par t_max, il faudrait
            réinjecter t*_max dans le résidu PDE (erreur fréquente).

        Parameters
        ----------
        t : float
            Temps en secondes, typiquement dans [0, t_max].

        Returns
        -------
        float
            t* ∈ [0, t*_max].
        """
        return t / self.t_ref

    def from_t_star(self, t_star: float) -> float:
        """
        Conversion inverse : t* (Fourier) → t (secondes).

            t = t* · t_ref
        """
        return t_star * self.t_ref

    def to_T_star(self, temperature: float) -> float:
        """
        Convertit une température (°C) en température adimensionnée.

            T* = (T - T_amb) / (T_obj - T_amb) = (T - T_amb) / delta_T

        Conséquences directes (à connaître par cœur) :
            T = T_amb  →  T* = 0
            T = T_obj  →  T* = 1
        """
        return (temperature - self.T_amb) / self.delta_T

    def from_T_star(self, T_star: float) -> float:
        """
        Conversion inverse : T* → T (°C).

            T = T* · delta_T + T_amb
        """
        return T_star * self.delta_T + self.T_amb

    # ==================================================================
    # Géométrie de l'objet chaud
    # ==================================================================

    def point_in_object(self, x_star: float, y_star: float) -> bool:
        """
        Test d'appartenance d'un point (x*, y*) à l'objet chaud.

        POURQUOI c'est important :
            À t* = 0, la condition initiale impose T* = 1 DANS l'objet
            et T* = 0 DEHORS. Ce test est la traduction géométrique
            de cette discontinuité initiale.

        Parameters
        ----------
        x_star, y_star : float
            Coordonnées adimensionnées du point à tester.

        Returns
        -------
        bool
            True  → le point est dans l'objet  → T* doit valoir 1
            False → le point est hors de l'objet → T* doit valoir 0
        """
        # Vecteur du centre de l'objet vers le point testé
        delta_x = x_star - self.obj_cx
        delta_y = y_star - self.obj_cy

        if self.obj_shape == "disk":
            # Disque : distance euclidienne ≤ rayon
            distance_squared = delta_x * delta_x + delta_y * delta_y
            return distance_squared <= self.obj_radius**2

        if self.obj_shape == "square":
            # Carré aligné axes : norme infinie ≤ demi-côté
            return (abs(delta_x) <= self.obj_radius) and (
                abs(delta_y) <= self.obj_radius
            )

        raise ValueError(
            f"Forme d'objet inconnue : {self.obj_shape!r}. "
            "Choisir 'disk' ou 'square'."
        )

    # ==================================================================
    # Affichage
    # ==================================================================

    def summary(self) -> str:
        """
        Résumé textuel multi-lignes de la configuration.

        Utile en début de script (main_step1) et en soutenance pour
        rappeler d'un coup d'œil toutes les échelles du problème.
        """
        object_description = (
            f"{self.obj_shape} @ ({self.obj_cx}, {self.obj_cy}), "
            f"r={self.obj_radius}"
        )
        lines = [
            "=" * 60,
            f"  {self.project_name} — DataConfig",
            "=" * 60,
            f"  Géométrie      : Lx={self.Lx} m, Ly={self.Ly} m",
            f"  Diffusivité    : α={self.alpha:.2e} m²/s",
            f"  Températures   : T_amb={self.T_amb}°C, T_obj={self.T_obj}°C",
            f"  ΔT             : {self.delta_T}°C",
            f"  Objet chaud    : {object_description}",
            f"  t_max          : {self.t_max} s  (t*_max = {self.t_star_max:.4f})",
            f"  t_ref = L²/α   : {self.t_ref:.1f} s",
            f"  Échantillonnage: N_ic={self.N_ic}, N_bc={self.N_bc}, N_res={self.N_res}",
            f"  Domaine *      : x*{self.x_star_range}, y*{self.y_star_range}, "
            f"t*={self.t_star_range}",
            f"  Seed / device  : {self.seed} / {self.device}",
            "=" * 60,
        ]
        return "\n".join(lines)


# Instance par défaut, importable partout :
#   from config import DEFAULT_CONFIG
# Évite de recréer DataConfig() à chaque appel si on ne customisera rien.
DEFAULT_CONFIG = DataConfig()
