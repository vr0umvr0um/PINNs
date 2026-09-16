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
from typing import Literal, Optional, Tuple

# Forme géométrique de l'objet chaud à t* = 0.
# "disk"   → disque de rayon obj_radius centré en (obj_cx, obj_cy)
# "square" → carré de demi-côté obj_radius centré idem
ObjectShape = Literal["disk", "square"]

# Fonctions d'activation autorisées pour le PINN.
# CONTRAINTE FONDAMENTALE : l'activation doit être DEUX FOIS dérivable
# (classe C²), car le résidu PDE contient ∂²T*/∂x*² et ∂²T*/∂y*².
# C'est pourquoi ReLU est volontairement ABSENTE de cette liste :
# sa dérivée seconde est identiquement nulle → le laplacien prédit
# vaudrait 0 partout et le PINN ne pourrait jamais apprendre la diffusion.
Activation = Literal["tanh", "sin", "gelu", "softplus"]


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


@dataclass
class ModelConfig:
    """
    Hyperparamètres d'ARCHITECTURE du réseau T_θ(x*, y*, t*)  — Étape 2.

    POURQUOI séparer de DataConfig ?
        DataConfig décrit la PHYSIQUE (elle ne change pas quand on teste
        un réseau plus profond). ModelConfig décrit le MODÈLE (il change
        à chaque expérience). Les garder distincts permet de balayer des
        architectures sans jamais toucher aux constantes physiques.
    """

    # ------------------------------------------------------------------
    # Topologie du MLP
    # ------------------------------------------------------------------
    # Entrée : 3 features (x*, y*, t*) — Sortie : 1 scalaire (T*).
    # Un MLP "4 couches × 64 neurones" est le point de départ classique
    # de la littérature PINN pour une EDP 2D+temps : assez expressif pour
    # capturer la diffusion, assez petit pour entraîner sur CPU.
    n_hidden_layers: int = 4  # nombre de couches cachées
    n_neurons: int = 64  # largeur de chaque couche cachée

    # ------------------------------------------------------------------
    # Non-linéarité
    # ------------------------------------------------------------------
    # tanh : choix de référence en PINN. Lisse (C^∞), bornée, et sa
    # dérivée seconde est non triviale → le laplacien autograd est correct.
    activation: Activation = "tanh"

    # Pulsation des activations sinusoïdales (utilisée seulement si
    # activation="sin", architecture de type SIREN). ω=30 est la valeur
    # recommandée par l'article SIREN pour des entrées normalisées [-1, 1].
    sine_omega: float = 30.0

    # ------------------------------------------------------------------
    # Normalisation des entrées
    # ------------------------------------------------------------------
    # POURQUOI normaliser alors qu'on a DÉJÀ adimensionné ?
    #     L'adimensionnement met x*, y* ∈ [0, 1] mais t* ∈ [0, 0.1].
    #     Le temps est donc 10× plus "petit" que l'espace : le réseau
    #     verrait une entrée quasi constante et les poids associés à t*
    #     recevraient des gradients ridiculement faibles.
    #     On applique donc une transformation affine FIXE (pas apprise)
    #     qui envoie chaque entrée sur [-1, 1], la zone où tanh est
    #     la plus sensible (dérivée maximale en 0).
    #
    # IMPORTANT (piège autograd) : cette normalisation interne ne change
    # RIEN au calcul du résidu. Autograd dérive la sortie par rapport au
    # tenseur t* FOURNI en entrée, et applique la règle de la chaîne à
    # travers la normalisation. On obtient bien ∂T*/∂t* au sens de Fourier.
    normalize_inputs: bool = True

    # ------------------------------------------------------------------
    # Initialisation des poids
    # ------------------------------------------------------------------
    # Glorot/Xavier : maintient la variance du signal constante d'une
    # couche à l'autre → évite l'explosion/extinction des gradients
    # dès l'initialisation, ce qui est critique quand on dérive DEUX fois.
    init_gain: float = 1.0

    # Graine dédiée à l'initialisation des poids (indépendante de celle
    # de l'échantillonnage → on peut rejouer le même nuage de points
    # avec un tirage de poids différent, et inversement).
    seed: int = 42

    def summary(self) -> str:
        """Résumé textuel de l'architecture (affiché par main_step2)."""
        lines = [
            "=" * 60,
            "  ModelConfig — architecture T_θ(x*, y*, t*)",
            "=" * 60,
            f"  Entrées        : 3 (x*, y*, t*)  →  Sortie : 1 (T*)",
            f"  Couches cachées: {self.n_hidden_layers} × {self.n_neurons} neurones",
            f"  Activation     : {self.activation}"
            + (f" (ω={self.sine_omega})" if self.activation == "sin" else ""),
            f"  Normalisation  : {'[-1, 1] affine fixe' if self.normalize_inputs else 'désactivée'}",
            f"  Init           : Xavier/Glorot (gain={self.init_gain}), seed={self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)


@dataclass
class LossConfig:
    """
    Pondérations de la loss multi-objectif du PINN — Étape 2.

        L(θ) = w_ic · L_ic  +  w_bc · L_bc  +  w_res · L_res

    POURQUOI pondérer ?
        Les trois termes n'ont ni la même échelle ni la même difficulté.
        L_ic doit capturer un CRÉNEAU discontinu (l'objet chaud) : c'est
        le terme le plus raide. L_res porte sur des dérivées secondes,
        numériquement plus bruitées. Sans pondération, l'optimiseur peut
        « satisfaire » le résidu avec la solution triviale T* ≡ 0, qui
        vérifie parfaitement l'EDP et les BC, mais rate l'IC.

    NOTE POUR L'ÉTAPE 3
        On garde ici des poids STATIQUES et neutres (1/1/1) : l'Étape 2
        ne fait que DÉFINIR la loss, pas l'optimiser. La pondération
        dynamique (et le Residual Adaptive Resampling) sont explicitement
        au programme de l'Étape 3.
    """

    w_ic: float = 1.0  # poids de la condition initiale
    w_bc: float = 1.0  # poids des conditions aux limites (Dirichlet)
    w_res: float = 1.0  # poids du résidu PDE

    def as_tuple(self) -> Tuple[float, float, float]:
        """Triplet (w_ic, w_bc, w_res), pratique pour le logging."""
        return (self.w_ic, self.w_bc, self.w_res)

    def summary(self) -> str:
        """Résumé textuel des pondérations."""
        return (
            f"LossConfig : L = {self.w_ic}·L_ic + "
            f"{self.w_bc}·L_bc + {self.w_res}·L_res"
        )


# ==========================================================================
# Étape 3 — stratégies d'entraînement
# ==========================================================================

# Stratégie de pondération des trois termes de la loss.
#   "fixed"        : poids statiques de LossConfig (référence, reproductible).
#   "grad_norm"    : w_i ∝ 1/‖∇_θ L_i‖₂  — équilibre les NORMES DE GRADIENTS
#                    des trois termes (inspiré de GradNorm, Chen et al. 2018,
#                    appliqué aux PINNs). Le terme dominant (ici L_res,
#                    cf. bilan de l'Étape 2) est dégonflé, les termes
#                    étouffés (L_ic, L_bc) sont remontés.
#   "lr_annealing" : Wang, Teng & Perdikaris (ICML 2021) — statistique
#                    max|∇_θ L_i| / mean|∇_θ L_i| calculée sur la PREMIÈRE
#                    couche : elle mesure le caractère « pointu » (mal
#                    conditionné) du gradient de chaque terme ; le pas
#                    effectif du terme pathologique est annealé vers le bas.
#
# Dans les deux schémas adaptatifs, les facteurs sont normalisés pour que
# le plus grand vale 1 (les poids restent bornés, la loss totale ne
# divergent pas en échelle), puis MULTIPLIÉS par les poids statiques de
# LossConfig : w_effectif = w_config × w_adaptatif.
WeightingScheme = Literal["fixed", "grad_norm", "lr_annealing"]

# Scheduler du learning rate pour la phase Adam.
#   "plateau" : ReduceLROnPlateau — divise le lr quand la loss stagne.
#   "cosine"  : CosineAnnealingLR — décroissance douce jusqu'à min_lr.
#   "none"    : lr constant.
SchedulerKind = Literal["plateau", "cosine", "none"]

# Métrique surveillée par l'early stopping et par le checkpoint "best".
#   "total"    : perte totale pondérée (l'objectif réellement minimisé).
#   "residual" : résidu EDP seul ( pertinent si l'on veut avant tout une
#               solution physiquement fidèle, au prix des BC/IC).
EarlyStopMonitor = Literal["total", "residual"]


@dataclass
class TrainConfig:
    """
    Hyperparamètres de la BOUCLE D'ENTRAÎNEMENT hybride — Étape 3.

        Phase 1 : Adam (exploration robuste, scheduler de lr)
        Phase 2 : L-BFGS (affinage quasi-Newton, line search strong_wolfe)

    POURQUOI DEUX PHASES ADAM → L-BFGS ?
        Adam normalise son pas par une moyenne mobile des gradients : il
        tolère les gradients raides des dérivées secondes du résidu et
        s'échappe des mauvais bassins — mais il OSCILLE indéfiniment
        autour du minimum sans jamais s'y installer (plafond ~1e-4/1e-5
        sur la loss). L-BFGS reconstruit une approximation inverse de la
        Hessienne à partir des dernières paires (pas, Δgradient) : son pas
        devient de plus en plus pertinent près du minimum → convergence
        quasi exacte. En contrepartie, loin du minimum son modèle de
        courbure est trompeur. D'où la séquence Adam PUIS L-BFGS —
        jamais l'inverse.

    POURQUOI FULL-BATCH (aucun mini-batch) ?
        L-BFGS mémorise une courbure ENTRE deux évaluations de la loss.
        Si l'objectif changeait entre deux appels de closure (mini-batchs
        aléatoires), la mémoire de courbure deviendrait incohérente et la
        line search échouerait. Tout le budget (N_ic, N_bc, N_res) est
        donc évalué à chaque itération ; l'entraînement est de plus
        parfaitement déterministe à seed fixée.

    POURQUOI PAS D'AMP (mixed precision) ?
        Le résidu exige une DOUBLE dérivation (backward du backward).
        En float16, les dérivées secondes saturent et produisent des NaN
        systématiques ; torch.autograd.grad n'est de toute façon pas
        couvert par GradScaler. Le projet reste en float32 : le modèle
        (~13k paramètres) est petit, le gain AMP serait négligeable.

    NOTE SUR LA PONDÉRATION DYNAMIQUE PENDANT L-BFGS
        L-BFGS minimise un objectif FIXE : changer w_ic/w_bc/w_res en
        cours de phase reviendrait à déplacer la cible sous les pieds de
        l'optimiseur et invaliderait sa mémoire de courbure. Les schémas
        adaptatifs s'appliquent donc pendant la phase Adam UNIQUEMENT ;
        la phase L-BFGS hérite des poids effectifs du dernier pas Adam.
    """

    # ------------------------------------------------------------------
    # Phase 1 — Adam (exploration)
    # ------------------------------------------------------------------
    # 5 000 epochs est un bon compromis CPU/GPU pour ce problème : assez
    # pour installer la structure de la solution, sans gaspiller du temps
    # là où L-BFGS sera de toute façon bien plus efficace.
    adam_epochs: int = 5000
    adam_lr: float = 1e-3  # lr nominal d'Adam (1e-3 : valeur par défaut robuste)
    # Clipping global optionnel de la norme du gradient (None = désactivé ;
    # Xavier + tanh rendent le clipping généralement inutile ici).
    adam_clip_grad_norm: Optional[float] = None

    # Scheduler de lr : ReduceLROnPlateau par défaut, car on ne connaît
    # pas d'avance le "bon" nombre d'epochs — il s'adapte à la loss.
    scheduler: SchedulerKind = "plateau"
    scheduler_factor: float = 0.5  # division du lr en cas de stagnation
    scheduler_patience: int = 500  # epochs sans améliration avant division
    scheduler_min_lr: float = 1e-6  # plancher du lr

    # ------------------------------------------------------------------
    # Phase 2 — L-BFGS (affinage)
    # ------------------------------------------------------------------
    # "itération" L-BFGS = UN appel à optimizer.step(closure), qui peut
    # lui-même évaluer la closure jusqu'à max_iter fois (line search).
    lbfgs_iterations: int = 500  # appels externes à step()
    lbfgs_lr: float = 1.0  # pas initial (les quasi-Newton s'autocalibrent)
    lbfgs_max_iter: int = 20  # itérations internes max par appel à step()
    lbfgs_history_size: int = 50  # paires (s, y) mémorisées pour la courbure
    lbfgs_tolerance_grad: float = 1e-9  # ‖∇L‖ < tol → arrêt interne
    lbfgs_tolerance_change: float = 1e-12  # |Δf| < tol → arrêt interne
    lbfgs_line_search_fn: Optional[Literal["strong_wolfe"]] = "strong_wolfe"

    # ------------------------------------------------------------------
    # Pondération dynamique des pertes
    # ------------------------------------------------------------------
    weighting: WeightingScheme = "fixed"
    # Mise à jour des facteurs adaptatifs toutes les N epochs Adam
    # (chaque mise à jour coûte 3 backward supplémentaires : garder N ≥ 50
    # pour un surcoût < 5 %).
    weighting_update_every: int = 100
    # Epochs Adam pendant lesquelles les poids restent ceux de la config
    # (laisser les transitoires d'initialisation passer).
    weighting_warmup: int = 0
    # Garde-fou numérique des divisions (normes de gradients quasi nulles).
    weighting_eps: float = 1e-8

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------
    # Filet de sécurité budgétaire : si la métrique surveillée n'améliore
    # pas de plus de es_min_delta pendant es_patience itérations
    # CONSÉCUTIVES SANS RECORD (Adam + L-BFGS cumulés), on stoppe net.
    early_stopping: bool = True
    es_monitor: EarlyStopMonitor = "total"
    es_patience: int = 1500
    es_min_delta: float = 0.0  # amélioration minimale absolue pour compter

    # ------------------------------------------------------------------
    # Logging & checkpoints
    # ------------------------------------------------------------------
    log_every: int = 100  # affichage terminal : 1 epoch Adam sur log_every
    lbfgs_log_every: int = 10  # affichage : 1 itération L-BFGS sur N
    checkpoint_every: int = 1000  # écriture de last_model.pt toutes les N itérations

    # ==================================================================
    # Validation & affichage
    # ==================================================================

    def __post_init__(self) -> None:
        """
        Vérifie la cohérence des hyperparamètres.

        POURQUOI valider ici plutôt qu'au milieu de l'entraînement ?
            Une erreur d'hyperparamètre détectée à l'epoch 3 000 gaspille
            des heures de calcul. Tout contrôler à la construction
            garantit l'échec immédiat, bruyant et explicite.
        """
        if self.adam_epochs < 0:
            raise ValueError(f"adam_epochs doit être ≥ 0, reçu {self.adam_epochs}.")
        if self.lbfgs_iterations < 0:
            raise ValueError(
                f"lbfgs_iterations doit être ≥ 0, reçu {self.lbfgs_iterations}."
            )
        if self.adam_lr <= 0.0 or self.lbfgs_lr <= 0.0:
            raise ValueError(
                f"Les lr doivent être > 0, reçu adam_lr={self.adam_lr}, "
                f"lbfgs_lr={self.lbfgs_lr}."
            )
        if self.scheduler not in ("plateau", "cosine", "none"):
            raise ValueError(f"Scheduler inconnu : {self.scheduler!r}.")
        if not (0.0 < self.scheduler_factor < 1.0):
            raise ValueError(
                f"scheduler_factor doit être dans (0, 1), reçu {self.scheduler_factor}."
            )
        if self.scheduler_patience < 0:
            raise ValueError(f"scheduler_patience doit être ≥ 0.")
        if self.scheduler_min_lr < 0.0:
            raise ValueError("scheduler_min_lr doit être ≥ 0.")
        if self.weighting not in ("fixed", "grad_norm", "lr_annealing"):
            raise ValueError(f"Schéma de pondération inconnu : {self.weighting!r}.")
        if self.weighting_update_every < 1:
            raise ValueError("weighting_update_every doit être ≥ 1.")
        if self.weighting_warmup < 0:
            raise ValueError("weighting_warmup doit être ≥ 0.")
        if self.weighting_eps <= 0.0:
            raise ValueError("weighting_eps doit être > 0.")
        if self.es_monitor not in ("total", "residual"):
            raise ValueError(f"es_monitor inconnu : {self.es_monitor!r}.")
        if self.es_patience < 1:
            raise ValueError("es_patience doit être ≥ 1.")
        if self.es_min_delta < 0.0:
            raise ValueError("es_min_delta doit être ≥ 0.")
        if self.log_every < 1 or self.lbfgs_log_every < 1 or self.checkpoint_every < 1:
            raise ValueError("log_every, lbfgs_log_every et checkpoint_every doivent être ≥ 1.")
        if self.lbfgs_max_iter < 1:
            raise ValueError("lbfgs_max_iter doit être ≥ 1.")
        if self.lbfgs_history_size < 1:
            raise ValueError("lbfgs_history_size doit être ≥ 1.")
        if self.lbfgs_line_search_fn not in (None, "strong_wolfe"):
            raise ValueError(
                f"lbfgs_line_search_fn invalide : {self.lbfgs_line_search_fn!r}. "
                "Choisir None ou 'strong_wolfe'."
            )
        if self.adam_clip_grad_norm is not None and self.adam_clip_grad_norm <= 0.0:
            raise ValueError("adam_clip_grad_norm doit être > 0 (ou None).")

    @property
    def lbfgs_max_closure_evals(self) -> int:
        """
        Borne supérieure du nombre d'évaluations de closure en phase L-BFGS.

        Chaque appel à step() peut évaluer la closure jusqu'à
        lbfgs_max_iter fois → budget pire-cas = iterations × max_iter.
        Utile pour estimer le coût d'un run avant de le lancer.
        """
        return self.lbfgs_iterations * self.lbfgs_max_iter

    def summary(self) -> str:
        """Résumé textuel de la boucle d'entraînement (affiché par main_step3)."""
        scheduler_description = {
            "plateau": (
                f"ReduceLROnPlateau (×{self.scheduler_factor} après "
                f"{self.scheduler_patience} epochs, min {self.scheduler_min_lr:.1e})"
            ),
            "cosine": (
                f"CosineAnnealingLR (T_max={self.adam_epochs}, "
                f"min {self.scheduler_min_lr:.1e})"
            ),
            "none": "désactivé (lr constant)",
        }[self.scheduler]
        weighting_description = (
            f"{self.weighting}"
            if self.weighting == "fixed"
            else (
                f"{self.weighting} (toutes les {self.weighting_update_every} epochs, "
                f"warmup {self.weighting_warmup}, × poids config)"
            )
        )
        lines = [
            "=" * 60,
            "  TrainConfig — boucle hybride Adam → L-BFGS",
            "=" * 60,
            f"  Phase 1 Adam   : {self.adam_epochs} epochs, lr={self.adam_lr:.1e}"
            + (f", clip={self.adam_clip_grad_norm}" if self.adam_clip_grad_norm else ""),
            f"  Scheduler      : {scheduler_description}",
            f"  Phase 2 L-BFGS : {self.lbfgs_iterations} itérations × "
            f"(max_iter={self.lbfgs_max_iter}, history={self.lbfgs_history_size})",
            f"                   lr={self.lbfgs_lr:.1f}, line search="
            f"{self.lbfgs_line_search_fn or 'aucune'}",
            f"  Pondération    : {weighting_description}",
            f"  Early stopping : {'activé' if self.early_stopping else 'désactivé'} "
            f"(métrique={self.es_monitor}, patience={self.es_patience}, "
            f"min_delta={self.es_min_delta:.1e})",
            f"  Logs           : 1/{self.log_every} epochs Adam, "
            f"1/{self.lbfgs_log_every} itérations L-BFGS",
            f"  Checkpoints    : toutes les {self.checkpoint_every} itérations "
            "+ fin de phase + records",
            "=" * 60,
        ]
        return "\n".join(lines)


# Instances par défaut, importables partout :
#   from config import DEFAULT_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_LOSS_CONFIG
#   from config import DEFAULT_TRAIN_CONFIG
# Évite de recréer les dataclasses à chaque appel si on ne customisera rien.
DEFAULT_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_LOSS_CONFIG = LossConfig()
DEFAULT_TRAIN_CONFIG = TrainConfig()
