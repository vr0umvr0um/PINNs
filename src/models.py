"""
==============================================================================
src/models.py — Architecture du réseau T_θ(x*, y*, t*)
==============================================================================

RÔLE DE CE MODULE (Étape 2)
---------------------------
Un PINN remplace le maillage d'un solveur classique par une FONCTION
PARAMÉTRIQUE continue :

    T_θ : (x*, y*, t*)  ∈ ℝ³   ⟶   T*  ∈ ℝ

Ici θ = les poids d'un perceptron multicouche (MLP). Une fois entraîné,
le réseau est un « champ de température analytique » : on peut l'évaluer
en N'IMPORTE quel point (x*, y*, t*), sans interpolation, et le DÉRIVER
exactement via autograd.

POURQUOI UN SIMPLE MLP SUFFIT-IL ?
----------------------------------
La solution de l'équation de la chaleur est très régulière (C^∞ dès que
t > 0 : la diffusion lisse instantanément la condition initiale). Un MLP
dense capture bien ce genre de champ lisse. Pas besoin de convolutions :
on n'a pas de grille, on a des points épars (x*, y*, t*).

LES DEUX CONTRAINTES NON NÉGOCIABLES DE L'ARCHITECTURE
------------------------------------------------------
1. **Activation de classe C²**
   Le résidu PDE contient ∂²T*/∂x*² et ∂²T*/∂y*². Si l'activation a une
   dérivée seconde nulle (ReLU, Leaky-ReLU) ou non définie, le laplacien
   calculé par autograd vaut 0 presque partout : le réseau « satisfait »
   alors ∂T*/∂t* = 0, c'est-à-dire un champ gelé. C'est LE bug classique.
   → d'où la liste blanche `Activation` dans config.py (tanh, sin, gelu, softplus).

2. **Entrées à la même échelle**
   Après adimensionnement : x*, y* ∈ [0, 1] mais t* ∈ [0, 0.1].
   Le temps est 10× plus « petit » : les poids de la colonne t*
   reçoivent des gradients 10× plus faibles et le réseau apprend un champ
   quasi stationnaire. On applique donc une normalisation affine FIXE
   vers [-1, 1] (voir `InputNormalization`).

Convention de shapes (identique à l'Étape 1)
--------------------------------------------
    entrées  : trois tenseurs (N, 1)  — x_star, y_star, t_star
    interne  : concaténation (N, 3)   — une ligne = un point de collocation
    sortie   : un tenseur (N, 1)      — T* prédit
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import (
    DEFAULT_MODEL_CONFIG,
    Activation,
    DataConfig,
    ModelConfig,
)


# ==========================================================================
# Briques élémentaires
# ==========================================================================

class Sine(nn.Module):
    """
    Activation sinusoïdale  x ↦ sin(ω x)  (architecture SIREN).

    POURQUOI la proposer à côté de tanh ?
        La condition initiale du problème est un CRÉNEAU (objet chaud à
        T*=1 dans un fond à T*=0). C'est une discontinuité, donc un
        contenu haute fréquence que tanh a du mal à représenter
        (phénomène de « spectral bias » : les MLP apprennent d'abord
        les basses fréquences). Les activations sinusoïdales atténuent
        ce biais. On la garde comme levier pour l'Étape 3.

    Comme tanh, sin est C^∞ : ses dérivées secondes sont bien définies,
    donc elle est compatible avec le résidu PDE.
    """

    def __init__(self, omega: float = 30.0) -> None:
        """
        Parameters
        ----------
        omega : float
            Pulsation. ω=30 est la valeur recommandée par l'article SIREN
            pour des entrées normalisées dans [-1, 1].
        """
        super().__init__()
        self.omega = float(omega)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applique sin(ω · x) élément par élément (shape inchangée)."""
        return torch.sin(self.omega * inputs)

    def extra_repr(self) -> str:
        """Affiche ω dans le print(model)."""
        return f"omega={self.omega}"


def build_activation(name: Activation, sine_omega: float = 30.0) -> nn.Module:
    """
    Fabrique le module d'activation correspondant à `name`.

    Toutes les options sont DEUX FOIS DÉRIVABLES, condition nécessaire
    pour que ∂²T*/∂x*² calculé par autograd ne soit pas identiquement nul.

    Parameters
    ----------
    name : {'tanh', 'sin', 'gelu', 'softplus'}
        Identifiant de l'activation (cf. `Activation` dans config.py).
    sine_omega : float
        Pulsation, utilisée uniquement si name == 'sin'.

    Returns
    -------
    nn.Module
        Module d'activation prêt à être inséré dans le nn.Sequential.

    Raises
    ------
    ValueError
        Si `name` est inconnue — en particulier 'relu', explicitement
        refusée car sa dérivée seconde est nulle presque partout.
    """
    if name == "tanh":
        return nn.Tanh()
    if name == "sin":
        return Sine(omega=sine_omega)
    if name == "gelu":
        return nn.GELU()
    if name == "softplus":
        return nn.Softplus()

    raise ValueError(
        f"Activation inconnue ou incompatible PINN : {name!r}. "
        "Choisir 'tanh', 'sin', 'gelu' ou 'softplus'. "
        "ReLU est exclue : sa dérivée seconde est nulle presque partout, "
        "donc le laplacien ∂²T*/∂x*² + ∂²T*/∂y*² serait identiquement 0."
    )


class InputNormalization(nn.Module):
    """
    Transformation affine FIXE (non apprise) des entrées vers [-1, 1].

        z = 2 · (v - lower) / (upper - lower) - 1

    POURQUOI c'est nécessaire ici
    -----------------------------
    Domaines adimensionnés du projet :
        x* ∈ [0, 1]      y* ∈ [0, 1]      t* ∈ [0, 0.1]
    Le temps occupe une plage 10× plus étroite. Sans recentrage :
        - la colonne t* de la première couche voit une entrée quasi nulle,
        - ses gradients sont ~10× plus petits que ceux de x*/y*,
        - le réseau converge vers un champ quasi stationnaire.

    POURQUOI [-1, 1] et pas [0, 1]
    -------------------------------
    tanh est centrée en 0 et y atteint sa pente maximale (tanh'(0) = 1).
    Centrer les entrées place donc le réseau dans son régime le plus
    expressif dès l'initialisation.

    POURQUOI des BUFFERS et pas des attributs Python
    -------------------------------------------------
    `register_buffer` fait suivre ces tenseurs lors des appels `.to(device)`
    et les inclut dans le `state_dict()`. À l'Étape 3 (checkpoints), un
    modèle rechargé retrouvera donc AUTOMATIQUEMENT les mêmes bornes de
    normalisation — sinon les prédictions seraient silencieusement fausses.

    PIÈGE AUTOGRAD (à savoir expliquer en soutenance)
    --------------------------------------------------
    Cette normalisation est INTERNE au réseau. Autograd dérive la sortie
    par rapport aux tenseurs (x*, y*, t*) fournis à `forward`, et remonte
    la règle de la chaîne à travers la normalisation. Le résidu calculé
    est donc bien exprimé dans les variables de Fourier, PAS dans les
    variables normalisées. Aucun facteur correctif à appliquer à la main.
    """

    def __init__(
        self,
        lower: Tuple[float, float, float],
        upper: Tuple[float, float, float],
    ) -> None:
        """
        Parameters
        ----------
        lower, upper : tuple of 3 floats
            Bornes (min, max) de chaque entrée, dans l'ordre (x*, y*, t*).
            Typiquement lower=(0, 0, 0) et upper=(1, 1, t*_max).
        """
        super().__init__()

        lower_tensor = torch.tensor(lower, dtype=torch.float32)
        upper_tensor = torch.tensor(upper, dtype=torch.float32)

        span = upper_tensor - lower_tensor
        if bool((span <= 0).any()):
            raise ValueError(
                f"Bornes de normalisation invalides : upper doit être "
                f"strictement supérieur à lower, reçu lower={lower}, upper={upper}."
            )

        # Shapes (1, 3) → broadcast direct sur un batch (N, 3)
        self.register_buffer("lower", lower_tensor.unsqueeze(0))
        self.register_buffer("span", span.unsqueeze(0))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        points : torch.Tensor of shape (N, 3)
            Colonnes (x*, y*, t*) en unités adimensionnées « physiques ».

        Returns
        -------
        torch.Tensor of shape (N, 3)
            Mêmes points, chaque colonne ramenée dans [-1, 1].
        """
        return 2.0 * (points - self.lower) / self.span - 1.0

    def extra_repr(self) -> str:
        """Affiche les bornes effectives dans le print(model)."""
        lower = self.lower.flatten().tolist()
        upper = (self.lower + self.span).flatten().tolist()
        return f"lower={lower}, upper={upper}"


# ==========================================================================
# Le réseau PINN
# ==========================================================================

class PINN(nn.Module):
    """
    MLP approximant le champ de température adimensionné T*(x*, y*, t*).

    Structure
    ---------
        (x*, y*, t*)                      3 tenseurs (N, 1)
             │  concaténation
             ▼
        points (N, 3)
             │  InputNormalization  →  [-1, 1]³   (optionnelle, fixe)
             ▼
        Linear(3 → H) → act → [Linear(H → H) → act] × (L-1) → Linear(H → 1)
             ▼
        T* prédit (N, 1)

    POURQUOI PAS DE BATCHNORM / DROPOUT
    ------------------------------------
    - BatchNorm rendrait la sortie dépendante des AUTRES points du batch :
      T_θ ne serait plus une fonction de (x*, y*, t*) seul, et la notion
      même de dérivée partielle ∂T*/∂x* perdrait son sens.
    - Dropout injecte du bruit stochastique : le résidu PDE deviendrait
      aléatoire d'un passage à l'autre, empêchant toute convergence fine.
    Un PINN est un problème d'AJUSTEMENT DE FONCTION, pas de généralisation
    statistique : la régularisation classique est contre-productive.

    Exemple
    -------
    >>> from config import DataConfig
    >>> from src.models import PINN
    >>> cfg = DataConfig()
    >>> model = PINN.from_config(cfg)
    >>> x = torch.rand(10, 1, requires_grad=True)
    >>> y = torch.rand(10, 1, requires_grad=True)
    >>> t = torch.rand(10, 1, requires_grad=True) * cfg.t_star_max
    >>> model(x, y, t).shape
    torch.Size([10, 1])
    """

    def __init__(
        self,
        n_hidden_layers: int = 4,
        n_neurons: int = 64,
        activation: Activation = "tanh",
        sine_omega: float = 30.0,
        normalize_inputs: bool = True,
        input_lower: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        input_upper: Tuple[float, float, float] = (1.0, 1.0, 0.1),
        init_gain: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_hidden_layers : int
            Nombre de couches cachées (≥ 1).
        n_neurons : int
            Largeur de chaque couche cachée.
        activation : {'tanh', 'sin', 'gelu', 'softplus'}
            Non-linéarité, obligatoirement C² (cf. build_activation).
        sine_omega : float
            Pulsation SIREN, utilisée seulement si activation='sin'.
        normalize_inputs : bool
            Active la normalisation affine des entrées vers [-1, 1].
        input_lower, input_upper : tuple of 3 floats
            Bornes (x*, y*, t*) utilisées par la normalisation.
            Ignorées si normalize_inputs=False.
        init_gain : float
            Gain de l'initialisation Xavier/Glorot.
        seed : int, optional
            Graine appliquée juste avant l'initialisation des poids,
            pour rendre le tirage initial reproductible.
        """
        super().__init__()

        if n_hidden_layers < 1:
            raise ValueError(
                f"n_hidden_layers doit valoir au moins 1, reçu {n_hidden_layers}."
            )

        # Mémorisé pour summary() / reproductibilité
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.sine_omega = sine_omega
        self.init_gain = init_gain

        # --- Normalisation des entrées (module ou identité) -------------
        self.normalizer: Optional[InputNormalization] = (
            InputNormalization(lower=input_lower, upper=input_upper)
            if normalize_inputs
            else None
        )

        # --- Empilement des couches -------------------------------------
        # 3 entrées (x*, y*, t*) → H → … → H → 1 sortie (T*)
        layers: list[nn.Module] = []

        # Première couche : projection de l'espace physique vers l'espace latent
        layers.append(nn.Linear(3, n_neurons))
        layers.append(build_activation(activation, sine_omega))

        # Couches cachées intermédiaires
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(build_activation(activation, sine_omega))

        # Couche de sortie : LINÉAIRE, sans activation.
        # POURQUOI pas de sigmoïde alors que T* ∈ [0, 1] physiquement ?
        #     Une sigmoïde finale saturerait (gradients ~0) dès que le
        #     réseau approche 0 ou 1, exactement là où vivent l'IC et les
        #     BC. On laisse la sortie libre : c'est la loss qui enseigne
        #     les bornes, pas l'architecture.
        layers.append(nn.Linear(n_neurons, 1))

        self.network = nn.Sequential(*layers)

        # --- Initialisation des poids -----------------------------------
        if seed is not None:
            torch.manual_seed(seed)
        self.apply_xavier_initialization()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def apply_xavier_initialization(self) -> None:
        """
        Initialise toutes les couches linéaires en Xavier/Glorot normal.

        POURQUOI Xavier plutôt que l'init PyTorch par défaut (Kaiming
        uniforme, pensée pour ReLU) ?
            Xavier conserve la variance du signal à la traversée d'une
            couche pour des activations SYMÉTRIQUES et centrées comme
            tanh. C'est d'autant plus critique ici qu'on dérive DEUX fois
            la sortie : une mauvaise init fait exploser ou annuler le
            laplacien dès le premier pas, et l'entraînement ne démarre pas.

        Les biais sont mis à zéro : aucune raison de privilégier une
        direction avant d'avoir vu le moindre point de collocation.
        """
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=self.init_gain)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Passe avant
    # ------------------------------------------------------------------

    def forward(
        self,
        x_star: torch.Tensor,
        y_star: torch.Tensor,
        t_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        Évalue T*_θ(x*, y*, t*).

        POURQUOI TROIS TENSEURS SÉPARÉS plutôt qu'un seul (N, 3) ?
            C'est LA signature qui rend l'autograd du PINN possible.
            `torch.autograd.grad` dérive par rapport à des tenseurs
            FEUILLES précis. En gardant x*, y*, t* séparés (chacun avec
            requires_grad=True, comme produit par l'Étape 1), on peut
            demander ∂T*/∂x* SANS toucher à ∂T*/∂t*.
            Si on passait un unique tenseur (N, 3), il faudrait ensuite
            découper le gradient colonne par colonne — faisable, mais
            plus fragile et moins lisible.

        Parameters
        ----------
        x_star, y_star, t_star : torch.Tensor of shape (N, 1)
            Coordonnées adimensionnées. Pour le calcul du résidu PDE,
            ces tenseurs doivent avoir requires_grad=True.

        Returns
        -------
        T_pred : torch.Tensor of shape (N, 1)
            Température adimensionnée prédite.
        """
        # Concaténation le long de la dimension des features :
        # 3 × (N, 1) → (N, 3). C'est ICI que le graphe autograd relie
        # la sortie aux trois tenseurs d'entrée.
        points = torch.cat([x_star, y_star, t_star], dim=1)  # (N, 3)

        if self.normalizer is not None:
            points = self.normalizer(points)  # (N, 3) ramené dans [-1, 1]

        return self.network(points)  # (N, 1)

    @torch.no_grad()
    def predict(
        self,
        x_star: torch.Tensor,
        y_star: torch.Tensor,
        t_star: torch.Tensor,
    ) -> torch.Tensor:
        """
        Évaluation SANS graphe autograd (inférence pure).

        À utiliser pour tracer une carte de chaleur ou alimenter le
        démonstrateur Gradio (Étape 4) : on ne veut ni dérivées ni
        consommation mémoire inutile.

        ATTENTION : ne JAMAIS utiliser predict() pour le résidu PDE.
        Sous `torch.no_grad()` le graphe n'est pas construit, donc
        `torch.autograd.grad` échouerait. Le résidu passe par forward().

        Returns
        -------
        T_pred : torch.Tensor of shape (N, 1), détaché du graphe.
        """
        return self.forward(x_star, y_star, t_star)

    # ------------------------------------------------------------------
    # Constructeurs alternatifs et introspection
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: DataConfig,
        model_cfg: Optional[ModelConfig] = None,
    ) -> "PINN":
        """
        Construit le réseau à partir des dataclasses de configuration.

        INTÉRÊT PRINCIPAL : les bornes de normalisation sont déduites
        automatiquement de la PHYSIQUE (cfg.t_star_max). Si on change
        alpha ou t_max dans DataConfig, la normalisation du temps suit
        toute seule — impossible d'oublier de la mettre à jour.

        Parameters
        ----------
        cfg : DataConfig
            Fournit x_star_range, y_star_range, t_star_max et device.
        model_cfg : ModelConfig, optional
            Hyperparamètres d'architecture. None → DEFAULT_MODEL_CONFIG.

        Returns
        -------
        PINN
            Réseau déjà déplacé sur cfg.device.
        """
        model_cfg = model_cfg or DEFAULT_MODEL_CONFIG

        model = cls(
            n_hidden_layers=model_cfg.n_hidden_layers,
            n_neurons=model_cfg.n_neurons,
            activation=model_cfg.activation,
            sine_omega=model_cfg.sine_omega,
            normalize_inputs=model_cfg.normalize_inputs,
            input_lower=(
                cfg.x_star_range[0],
                cfg.y_star_range[0],
                0.0,
            ),
            input_upper=(
                cfg.x_star_range[1],
                cfg.y_star_range[1],
                cfg.t_star_max,
            ),
            init_gain=model_cfg.init_gain,
            seed=model_cfg.seed,
        )
        return model.to(cfg.device)

    @property
    def n_parameters(self) -> int:
        """
        Nombre de paramètres ENTRAÎNABLES (poids + biais).

        Ordre de grandeur attendu pour 4×64 :
            3·64+64  +  3 × (64·64+64)  +  64·1+1  ≈ 12 8xx paramètres.
        C'est minuscule face à un réseau de vision — normal : on
        approxime une fonction lisse de 3 variables, pas des images.
        """
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def summary(self) -> str:
        """
        Résumé textuel compact de l'architecture (pour le terminal).

        Affiché par main_step2.py : permet de montrer d'un coup d'œil
        en soutenance la profondeur, la largeur, l'activation et le
        nombre de paramètres.
        """
        normalization_line = (
            "activée → [-1, 1]" if self.normalizer is not None else "désactivée"
        )
        lines = [
            "=" * 60,
            "  PINN — T_θ(x*, y*, t*)",
            "=" * 60,
            f"  Topologie      : 3 → "
            + " → ".join([str(self.n_neurons)] * self.n_hidden_layers)
            + " → 1",
            f"  Activation     : {self.activation_name} (C², compatible ∂²/∂x*²)",
            f"  Normalisation  : {normalization_line}",
            f"  Paramètres     : {self.n_parameters:,} entraînables",
            "=" * 60,
        ]
        return "\n".join(lines)


__all__ = [
    "PINN",
    "Sine",
    "InputNormalization",
    "build_activation",
]
