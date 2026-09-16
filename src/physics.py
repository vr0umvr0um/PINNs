"""
==============================================================================
src/physics.py — Dérivation automatique & résidu de l'EDP
==============================================================================

RÔLE DE CE MODULE (cœur de l'Étape 2)
--------------------------------------
C'est ici que la PHYSIQUE entre dans le réseau. Un réseau ordinaire
apprendrait T* à partir de mesures ; un PINN apprend T* à partir de
l'ÉQUATION elle-même, en pénalisant son résidu :

    r(x*, y*, t*)  =  ∂T*/∂t*  −  ( ∂²T*/∂x*²  +  ∂²T*/∂y*² )

Si r ≡ 0 partout, le réseau vérifie l'équation de la chaleur adimensionnée.
La loss résidu (losses.py) est simplement MSE(r, 0).

POURQUOI L'AUTOGRAD ET PAS DES DIFFÉRENCES FINIES ?
----------------------------------------------------
Un solveur classique approxime ∂²T/∂x² par (T_{i+1} − 2T_i + T_{i−1})/h² :
c'est une APPROXIMATION, dont l'erreur en O(h²) impose un maillage fin et
introduit de la diffusion numérique.

Ici T_θ est une fonction analytique composée d'opérations élémentaires
(produits matriciels, tanh). PyTorch connaît la dérivée EXACTE de chacune
et applique la règle de la chaîne : `torch.autograd.grad` renvoie la vraie
dérivée, à la précision machine près, en tout point et sans maillage.
C'est l'argument central du PINN, et le point à marteler en soutenance.

LE MÉCANISME EN TROIS TEMPS
---------------------------
    1. x*, y*, t* portent requires_grad=True     (fait dès l'Étape 1)
    2. T_pred = model(x*, y*, t*)                → construit le graphe
    3. grad(T_pred, x*) puis grad(∂T*/∂x*, x*)   → dérivée seconde

Le paramètre `create_graph=True` du premier appel est INDISPENSABLE :
sans lui, ∂T*/∂x* est un tenseur « mort », détaché du graphe, et le
second appel à grad() lève une erreur. C'est le piège n°1 du module.

VALIDATION INTÉGRÉE
-------------------
Le module fournit `analytic_solution`, un mode propre de l'équation de la
chaleur dont on connaît la solution exacte. En l'injectant à la place du
réseau, `pde_residual` doit renvoyer ~0 à la précision machine. C'est le
test qui prouve que l'implémentation autograd est correcte — bien plus
convaincant qu'un simple « ça tourne sans erreur ».
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Tuple

import torch

# Un « champ de température » est tout callable qui mappe trois tenseurs
# (N, 1) vers un tenseur (N, 1). Cela couvre :
#   - un modèle PINN (nn.Module est callable),
#   - une solution analytique (fonction Python),
#   - un champ factice construit pour un test.
# Typer large permet de VALIDER le résidu sans réseau entraîné.
TemperatureField = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# ==========================================================================
# Brique de base : dérivation par autograd
# ==========================================================================

def gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Dérive `outputs` par rapport à `inputs` via `torch.autograd.grad`.

    POURQUOI `grad_outputs=ones_like(outputs)` ?
        `torch.autograd.grad` calcule un produit vecteur-jacobienne
        (VJP) : il attend un vecteur v et renvoie vᵀ · J. Notre sortie
        est un tenseur (N, 1), pas un scalaire.
        En prenant v = 1 partout, on somme les dérivées sur le batch :

            [vᵀ J]_i = Σ_k ∂T_k / ∂x_i

        Or T_k ne dépend QUE de x_k (chaque point de collocation est
        traité indépendamment par le réseau) : tous les termes croisés
        ∂T_k/∂x_i avec k ≠ i sont nuls. Le résultat est donc exactement
        le vecteur des dérivées ponctuelles (∂T_i/∂x_i), shape (N, 1).
        C'est l'astuce qui permet de dériver TOUT le batch en un appel.

    POURQUOI `create_graph=True` ?
        Le gradient renvoyé reste alors RATTACHÉ au graphe de calcul.
        Deux conséquences, toutes deux vitales :
          1. on peut le redériver → dérivées secondes (le laplacien) ;
          2. la loss résidu redevient différentiable par rapport à θ,
             donc `loss.backward()` peut entraîner le réseau.
        Avec create_graph=False, on obtient une valeur numérique morte.

    Parameters
    ----------
    outputs : torch.Tensor of shape (N, 1)
        Quantité à dériver (T* prédit, ou une dérivée première).
    inputs : torch.Tensor of shape (N, 1)
        Variable de dérivation, avec requires_grad=True.
    create_graph : bool
        True (défaut) → résultat redérivable et entraînable.
        False → dérivée « détachée », uniquement pour du diagnostic.

    LE CAS DÉGÉNÉRÉ DU CHAMP AFFINE
        Si le champ est AFFINE en x — par exemple la solution triviale
        T* ≡ 0, ou un réseau dégénéré — alors ∂T*/∂x* est une CONSTANTE.
        Cette constante ne dépend plus de x : il n'existe plus aucun
        chemin dans le graphe permettant de la redériver, et PyTorch lève
        « does not require grad » au lieu de renvoyer la bonne réponse
        mathématique, à savoir ∂²T*/∂x*² = 0.

        Deux garde-fous couvrent les deux façons dont ce cas se présente :
          - `outputs` entièrement détaché du graphe (dérivée constante)
            → on renvoie directement un tenseur de zéros ;
          - `inputs` présent mais inutilisé par `outputs` (champ ignorant
            l'une des trois variables) → `allow_unused` + `materialize_grads`
            demandent à autograd de matérialiser des zéros.

        Le résidu d'un champ affine devient ainsi CALCULABLE, ce qui
        compte : la solution triviale T* ≡ 0 est précisément le piège que
        la loss multi-objectif doit éviter, et on veut pouvoir l'évaluer
        pour le démontrer.

    Returns
    -------
    torch.Tensor of shape (N, 1)
        Dérivée ponctuelle ∂outputs/∂inputs.

    Raises
    ------
    RuntimeError
        Si `inputs` n'a pas requires_grad=True, ou si le graphe reliant
        outputs à inputs n'existe pas (par exemple parce que la passe
        avant a été faite sous `torch.no_grad()`).
    """
    # Champ affine : `outputs` est une constante détachée du graphe.
    # Sa dérivée vaut 0 — autograd ne saurait pas le calculer faute de chemin.
    if not outputs.requires_grad:
        return torch.zeros_like(inputs)

    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,  # le graphe resservira pour les autres dérivées
        allow_unused=True,  # variable absente du champ → gradient nul
        materialize_grads=True,  # …que l'on matérialise alors à 0
    )[0]


# ==========================================================================
# Dérivées de l'équation de la chaleur
# ==========================================================================

def heat_derivatives(
    field: TemperatureField,
    x_star: torch.Tensor,
    y_star: torch.Tensor,
    t_star: torch.Tensor,
    create_graph: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Calcule T* et toutes ses dérivées utiles à l'équation de la chaleur.

    Dérivées produites
    ------------------
        T      : T*(x*, y*, t*)          (la prédiction elle-même)
        T_t    : ∂T*/∂t*                 (terme instationnaire)
        T_x    : ∂T*/∂x*                 (intermédiaire, = flux en x)
        T_y    : ∂T*/∂y*                 (intermédiaire, = flux en y)
        T_xx   : ∂²T*/∂x*²               (diffusion selon x)
        T_yy   : ∂²T*/∂y*²               (diffusion selon y)
        laplacian : T_xx + T_yy          (Δ*T*)

    POURQUOI EXPOSER LES INTERMÉDIAIRES ?
        - T_x et T_y sont proportionnels au FLUX de chaleur (loi de
          Fourier : q = −k ∇T). Les visualiser est très parlant.
        - Séparer T_xx et T_yy permet de vérifier la symétrie du
          problème : l'objet chaud étant centré, on doit retrouver
          des contributions équivalentes en x et en y.

    POURQUOI LE PREMIER ÉTAGE FORCE create_graph=True
        Les dérivées premières T_x et T_y ne sont pas une fin en soi :
        on doit pouvoir les REDÉRIVER pour obtenir T_xx et T_yy. Elles
        sont donc toujours calculées en gardant le graphe, quelle que
        soit la valeur de `create_graph`. Le paramètre ne pilote que le
        DERNIER étage (dérivées secondes + T_t), c'est-à-dire :
            True  → le résidu est différentiable par rapport à θ
                    (obligatoire pendant l'entraînement) ;
            False → le résidu est une valeur morte, moins coûteuse
                    en mémoire (diagnostic, validation, figures).

    Parameters
    ----------
    field : TemperatureField
        Modèle PINN ou fonction analytique (x*, y*, t*) → T*.
    x_star, y_star, t_star : torch.Tensor of shape (N, 1)
        Points de collocation, requires_grad=True obligatoire.
    create_graph : bool
        Cf. explication ci-dessus.

    Returns
    -------
    dict[str, torch.Tensor]
        Toutes les quantités ci-dessus, chacune de shape (N, 1).
    """
    for name, tensor in (
        ("x_star", x_star),
        ("y_star", y_star),
        ("t_star", t_star),
    ):
        if not tensor.requires_grad:
            raise RuntimeError(
                f"`{name}` doit avoir requires_grad=True pour dériver le "
                "résidu PDE. Les batches produits par src.sampling le font "
                "déjà ; si le tenseur vient d'ailleurs, appeler "
                f"{name}.requires_grad_(True) avant."
            )

    # --- Passe avant : construit le graphe reliant T aux entrées --------
    T = field(x_star, y_star, t_star)  # (N, 1)

    # --- Dérivées premières ---------------------------------------------
    # create_graph=True imposé : T_x et T_y doivent rester redérivables.
    T_t = gradient(T, t_star, create_graph=create_graph)
    T_x = gradient(T, x_star, create_graph=True)
    T_y = gradient(T, y_star, create_graph=True)

    # --- Dérivées secondes (le laplacien) --------------------------------
    T_xx = gradient(T_x, x_star, create_graph=create_graph)
    T_yy = gradient(T_y, y_star, create_graph=create_graph)

    return {
        "T": T,
        "T_t": T_t,
        "T_x": T_x,
        "T_y": T_y,
        "T_xx": T_xx,
        "T_yy": T_yy,
        "laplacian": T_xx + T_yy,
    }


def pde_residual(
    field: TemperatureField,
    x_star: torch.Tensor,
    y_star: torch.Tensor,
    t_star: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Résidu de l'équation de la chaleur adimensionnée.

        r  =  ∂T*/∂t*  −  ( ∂²T*/∂x*²  +  ∂²T*/∂y*² )

    POURQUOI AUCUN COEFFICIENT α DEVANT LE LAPLACIEN ?
        C'est tout le bénéfice de l'adimensionnement de l'Étape 1.
        En posant t* = α t / L_ref² (le nombre de Fourier), α est ABSORBÉ
        dans la variable de temps et l'équation devient universelle.
        Si l'on avait normalisé le temps par t_max au lieu de t_ref, il
        faudrait réintroduire un facteur ici — erreur classique qui rend
        le PINN silencieusement faux (il résout la bonne équation… avec
        la mauvaise diffusivité).

    POURQUOI LE SIGNE « − » ET PAS « + » ?
        Convention : on écrit l'EDP sous forme homogène F(T) = 0.
            ∂T/∂t = ΔT      ⟺      ∂T/∂t − ΔT = 0
        Le résidu est le membre de gauche. Son signe n'a en réalité
        aucune importance pour la loss (on prend le carré), mais il en a
        pour l'INTERPRÉTATION d'une carte de résidu : r > 0 signifie que
        le réseau chauffe trop vite par rapport à ce que la diffusion
        autorise.

    Parameters
    ----------
    field : TemperatureField
        Modèle PINN ou solution analytique.
    x_star, y_star, t_star : torch.Tensor of shape (N, 1)
        Points de collocation intérieurs, requires_grad=True.
    create_graph : bool
        True pendant l'entraînement (résidu différentiable en θ),
        False pour un simple diagnostic.

    Returns
    -------
    residual : torch.Tensor of shape (N, 1)
        Vaut ~0 là où l'équation est satisfaite.
    """
    derivatives = heat_derivatives(
        field, x_star, y_star, t_star, create_graph=create_graph
    )
    return derivatives["T_t"] - derivatives["laplacian"]


# ==========================================================================
# Solution analytique de référence (validation de l'autograd)
# ==========================================================================

def analytic_solution(
    x_star: torch.Tensor,
    y_star: torch.Tensor,
    t_star: torch.Tensor,
    modes: Tuple[int, int] = (1, 1),
) -> torch.Tensor:
    """
    Mode propre exact de l'équation de la chaleur sur le carré unité.

        T*(x*, y*, t*) = sin(m π x*) · sin(n π y*) · exp(−(m² + n²) π² t*)

    POURQUOI CETTE FONCTION EST PRÉCIEUSE ICI
    ------------------------------------------
    Elle vérifie EXACTEMENT l'équation cible. Vérification à la main :

        ∂T*/∂t*   = −(m² + n²) π² · T*
        ∂²T*/∂x*² = −m² π² · T*
        ∂²T*/∂y*² = −n² π² · T*
        ⟹  ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)
           = −(m²+n²)π² T*  + m²π² T* + n²π² T*  =  0     ✓

    Elle satisfait aussi les conditions de Dirichlet homogènes du
    problème (sin(mπ·0) = sin(mπ·1) = 0 : nulle sur les 4 parois).

    Deux usages :
      1. **Étape 2 (ici)** — l'injecter dans `pde_residual` à la place du
         réseau. Le résidu doit tomber à la précision machine. Si ce
         n'est pas le cas, le bug est dans l'autograd, pas dans le
         réseau ni dans l'entraînement. C'est un test décisif.
      2. **Étape 4** — servir de solution de référence pour mesurer une
         erreur L2 relative, en complément du solveur différences finies.

    ATTENTION : ce n'est PAS la solution du problème du sujet. La vraie
    condition initiale est un créneau (objet chaud), pas un sinus. C'est
    un outil de VALIDATION NUMÉRIQUE, pas la cible physique.

    Parameters
    ----------
    x_star, y_star, t_star : torch.Tensor of shape (N, 1)
        Coordonnées adimensionnées.
    modes : tuple of 2 ints
        Les entiers (m, n) du mode propre. (1, 1) = mode fondamental,
        le plus lisse et le plus lentement amorti.

    Returns
    -------
    torch.Tensor of shape (N, 1)
        Valeurs exactes de T* pour ce mode.
    """
    mode_x, mode_y = modes
    decay_rate = (mode_x**2 + mode_y**2) * math.pi**2

    return (
        torch.sin(mode_x * math.pi * x_star)
        * torch.sin(mode_y * math.pi * y_star)
        * torch.exp(-decay_rate * t_star)
    )


def relative_residual_error(
    field: TemperatureField,
    x_star: torch.Tensor,
    y_star: torch.Tensor,
    t_star: torch.Tensor,
) -> float:
    """
    Erreur RELATIVE du résidu, adimensionnée par l'échelle de ∂T*/∂t*.

        erreur = ‖ r ‖₂  /  ( ‖ ∂T*/∂t* ‖₂ + ε )

    POURQUOI RELATIVISER ?
        Un résidu « absolu » de 1e-4 ne veut rien dire tout seul : il est
        excellent si ∂T*/∂t* ~ 100, catastrophique si ∂T*/∂t* ~ 1e-4.
        En divisant par la norme du terme instationnaire, on obtient un
        nombre sans dimension directement comparable d'un cas à l'autre.

    Utilisé par main_step2 et les tests pour valider l'autograd sur la
    solution analytique : on doit obtenir ~1e-14 en float64.

    Returns
    -------
    float
        Erreur relative (sans dimension).
    """
    derivatives = heat_derivatives(
        field, x_star, y_star, t_star, create_graph=False
    )
    residual = derivatives["T_t"] - derivatives["laplacian"]

    residual_norm = float(torch.linalg.vector_norm(residual))
    scale_norm = float(torch.linalg.vector_norm(derivatives["T_t"]))

    return residual_norm / (scale_norm + 1e-30)


__all__ = [
    "gradient",
    "heat_derivatives",
    "pde_residual",
    "analytic_solution",
    "relative_residual_error",
    "TemperatureField",
]
