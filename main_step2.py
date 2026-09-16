#!/usr/bin/env python3
"""
==============================================================================
main_step2.py — Étape 2 : Architecture du PINN & Dérivation Automatique
==============================================================================

Rôle de ce script dans le projet
--------------------------------
Point d'entrée de l'Étape 2. Il ne fait PAS d'entraînement (c'est l'objet
de l'Étape 3) : il CONSTRUIT et VALIDE la machinerie qui rendra
l'entraînement possible.

    1. Instancier le réseau T_θ(x*, y*, t*) et afficher son architecture
    2. Vérifier la passe avant (shapes, dtypes, graphe autograd)
    3. VALIDER l'autograd — le point crucial de l'étape :
         a) sur une solution analytique exacte  → résidu ≈ 0 (précision machine)
         b) contre des différences finies       → dérivées du réseau correctes
         c) contre-exemple ReLU                 → laplacien nul, PINN impossible
    4. Évaluer la loss multi-objectif L = w_ic·L_ic + w_bc·L_bc + w_res·L_res
    5. Vérifier que le gradient ∂L/∂θ atteint bien TOUS les paramètres
    6. Exporter une figure diagnostic `step2_pinn.png`

Pourquoi valider l'autograd AVANT d'entraîner ?
    Un bug d'autograd ne plante pas : il produit silencieusement un
    résidu faux. Le réseau converge alors très bien… vers la solution
    d'une autre équation. Le seul moyen de s'en prémunir est de tester
    le résidu sur une fonction dont on connaît la réponse exacte.
    C'est exactement ce que fait l'étape 3a.

Usage
-----
    python main_step2.py
    python main_step2.py --layers 6 --neurons 128 --activation sin
    python main_step2.py --w-ic 10 --w-bc 10 --w-res 1
    python main_step2.py --n-ic 500 --n-bc 500 --n-res 2000   # run rapide
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.axes import Axes

from config import DataConfig, LossConfig, ModelConfig
from src.losses import LossTerms, pinn_loss
from src.models import PINN
from src.physics import (
    analytic_solution,
    gradient,
    heat_derivatives,
    pde_residual,
    relative_residual_error,
)
from src.sampling import sample_bc, sample_ic, sample_residual
from src.utils import set_seed

CollocationBatch = Dict[str, torch.Tensor]


# ==========================================================================
# 2. Validation de la passe avant
# ==========================================================================

def validate_forward(model: PINN, cfg: DataConfig) -> None:
    """
    Vérifie le contrat de `PINN.forward`.

    Invariants contrôlés
    --------------------
    1. Sortie de shape (N, 1) — une température par point de collocation.
    2. Sortie rattachée au graphe autograd (`grad_fn` non nul) : sans
       cela, aucune dérivée ne pourra être calculée.
    3. Sortie FINIE (ni NaN ni inf) dès l'initialisation. Un NaN ici
       signalerait une initialisation ou une normalisation cassée, et
       contaminerait irrémédiablement l'entraînement.
    4. Le réseau est bien une FONCTION du temps : deux instants
       différents doivent donner des prédictions différentes. C'est le
       garde-fou contre une normalisation temporelle qui écraserait t*.
    """
    n_points = 16
    x_star = torch.rand(n_points, 1, device=cfg.device, requires_grad=True)
    y_star = torch.rand(n_points, 1, device=cfg.device, requires_grad=True)
    t_star = torch.full(
        (n_points, 1), 0.5 * cfg.t_star_max, device=cfg.device
    ).requires_grad_(True)

    T_pred = model(x_star, y_star, t_star)

    expected_shape = (n_points, 1)
    actual_shape = tuple(T_pred.shape)
    assert actual_shape == expected_shape, (
        f"forward : shape {actual_shape} ≠ {expected_shape}"
    )
    assert T_pred.grad_fn is not None, (
        "forward : la sortie doit être rattachée au graphe autograd "
        "(vérifier qu'on n'est pas sous torch.no_grad())"
    )
    assert torch.isfinite(T_pred).all(), (
        "forward : la sortie contient des NaN/inf dès l'initialisation"
    )

    # Le réseau dépend-il réellement de t* ?
    t_other = torch.zeros_like(t_star).requires_grad_(True)
    T_other = model(x_star, y_star, t_other)
    time_sensitivity = float((T_pred - T_other).abs().max())
    assert time_sensitivity > 1e-8, (
        "forward : la prédiction est insensible à t*. La normalisation "
        "temporelle écrase probablement l'entrée."
    )

    print(
        f"  [OK] forward : sortie {actual_shape}, dtype={T_pred.dtype}, "
        f"grad_fn={type(T_pred.grad_fn).__name__}, "
        f"sensibilité à t* = {time_sensitivity:.3e}"
    )


# ==========================================================================
# 3a. Validation de l'autograd sur la solution analytique
# ==========================================================================

def validate_autograd_on_analytic(cfg: DataConfig, n_points: int = 4096) -> float:
    """
    LE test décisif de l'Étape 2.

    Principe
    --------
    On injecte dans `pde_residual` non pas le réseau, mais la solution
    EXACTE d'un mode propre de l'équation de la chaleur :

        T*(x*, y*, t*) = sin(πx*) sin(πy*) exp(−2π² t*)

    Cette fonction vérifie ∂T*/∂t* = Δ*T* de façon exacte. Si notre
    chaîne autograd est correcte, le résidu calculé doit tomber à la
    PRÉCISION MACHINE. Sinon, le bug est dans physics.py — pas dans
    l'architecture, pas dans l'entraînement.

    POURQUOI EN float64 ?
        Le projet travaille en float32 (bon compromis vitesse/mémoire),
        où la précision relative est ~1e-7. Les dérivées secondes
        amplifient cette erreur d'arrondi et le résidu plafonne vers
        1e-5 : correct, mais peu probant. En float64 (précision ~1e-16),
        on attend ~1e-13, ce qui ne laisse aucun doute sur la validité
        de l'implémentation.

    CONTRE-ÉPREUVE
        On évalue aussi le résidu sur un champ qui NE vérifie PAS
        l'équation : T = x*² + y*² + t*, pour lequel
            ∂T/∂t = 1,  ΔT = 2 + 2 = 4,  donc  r = 1 − 4 = −3.
        Le résidu doit alors valoir exactement −3. Sans cette
        contre-épreuve, un `pde_residual` qui renverrait bêtement zéro
        passerait le premier test avec succès.

    Returns
    -------
    float
        L'erreur relative obtenue sur la solution analytique.
    """
    set_seed(cfg.seed)

    # Points tirés en float64 dans le domaine physique
    x_star = torch.rand(n_points, 1, dtype=torch.float64, requires_grad=True)
    y_star = torch.rand(n_points, 1, dtype=torch.float64, requires_grad=True)
    t_star = (
        torch.rand(n_points, 1, dtype=torch.float64) * cfg.t_star_max
    ).requires_grad_(True)

    # --- (a) Le résidu doit s'annuler sur la solution exacte ------------
    relative_error = relative_residual_error(
        analytic_solution, x_star, y_star, t_star
    )
    assert relative_error < 1e-10, (
        f"Autograd : résidu relatif {relative_error:.3e} sur une solution "
        "EXACTE de l'équation de la chaleur. L'implémentation de "
        "pde_residual est incorrecte."
    )

    # Vérification supplémentaire des dérivées individuelles :
    # pour ce mode, ∂T*/∂t* doit valoir exactement −2π² · T*.
    derivatives = heat_derivatives(
        analytic_solution, x_star, y_star, t_star, create_graph=False
    )
    expected_T_t = -2.0 * (np.pi**2) * derivatives["T"]
    time_derivative_error = float(
        (derivatives["T_t"] - expected_T_t).abs().max()
    )
    assert time_derivative_error < 1e-9, (
        f"Autograd : ∂T*/∂t* incorrect (écart max {time_derivative_error:.3e} "
        "par rapport à la valeur analytique −2π²·T*)"
    )

    # --- (b) Contre-épreuve : un champ qui viole l'équation -------------
    def non_solution(
        x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """T = x² + y² + t  →  ∂T/∂t − ΔT = 1 − 4 = −3."""
        return x**2 + y**2 + t

    residual_violation = pde_residual(
        non_solution, x_star, y_star, t_star, create_graph=False
    )
    violation_error = float((residual_violation + 3.0).abs().max())
    assert violation_error < 1e-9, (
        f"Autograd : sur T = x²+y²+t le résidu devrait valoir −3, "
        f"écart max observé {violation_error:.3e}"
    )

    print(
        f"  [OK] analytique : résidu relatif = {relative_error:.3e} "
        f"(float64, N={n_points})  |  ∂T*/∂t* exact à {time_derivative_error:.1e}"
    )
    print(
        f"  [OK] contre-épreuve : T=x²+y²+t donne bien r = −3 "
        f"(écart {violation_error:.1e}) → le résidu n'est pas trivialement nul"
    )
    return relative_error


# ==========================================================================
# 3b. Validation contre les différences finies
# ==========================================================================

def _finite_difference_step(model: PINN) -> float:
    """
    Choisit le pas h des différences finies selon l'activation du réseau.

    POURQUOI LE PAS NE PEUT PAS ÊTRE UNIVERSEL
        Une différence finie centrée souffre de DEUX erreurs opposées :
            - troncature  ~ h² · T'''      → pousse à réduire h
            - arrondi     ~ ε_machine / h² → pousse à augmenter h
        L'optimum dépend donc de la « raideur » de la fonction, c'est-à-
        dire des fréquences que le réseau contient.

        tanh produit un champ doux : h = 1e-4 est optimal.
        SIREN (sin, ω=30) oscille ~30× plus vite à chaque couche ; à
        h = 1e-4 la fonction change plusieurs fois de sens sur
        l'intervalle, et c'est la RÉFÉRENCE qui devient fausse — pas
        l'autograd. Il faut descendre à h ≈ 1e-7.

    C'est d'ailleurs, en creux, l'argument central du PINN : l'autograd
    n'a aucun pas à régler et reste exact quelle que soit la fréquence
    du champ.
    """
    return 1e-7 if model.activation_name == "sin" else 1e-4


def validate_autograd_vs_finite_differences(
    model: PINN,
    cfg: DataConfig,
    n_points: int = 256,
    step: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Confronte les dérivées autograd du RÉSEAU à des différences finies.

    POURQUOI CE SECOND TEST, PUISQUE L'ANALYTIQUE EST DÉJÀ PASSÉ ?
        Le test analytique valide `physics.py` sur une fonction Python
        simple. Celui-ci valide la chaîne COMPLÈTE sur le vrai réseau :
        concaténation (N,1)×3 → (N,3), normalisation affine interne,
        couches linéaires, activations. En particulier, il détecterait
        une normalisation des entrées mal prise en compte — le piège
        où l'on obtiendrait ∂T*/∂t* exprimé dans la variable NORMALISÉE
        au lieu de la variable de Fourier (facteur 2/t*_max d'écart).

    Différences finies centrées utilisées :
        ∂T/∂x  ≈ [ T(x+h) − T(x−h) ] / (2h)                erreur O(h²)
        ∂²T/∂x² ≈ [ T(x+h) − 2T(x) + T(x−h) ] / h²          erreur O(h²)

    Le modèle est converti en float64 le temps du test : en float32,
    la soustraction T(x+h) − T(x−h) perdrait presque toute sa précision
    (annulation catastrophique), et le test échouerait pour de mauvaises
    raisons.

    Returns
    -------
    (erreur_derivee_premiere, erreur_derivee_seconde) : tuple of float
        Écarts relatifs max entre autograd et différences finies.
    """
    if step is None:
        step = _finite_difference_step(model)

    # Copie float64 : on ne touche pas au modèle de travail
    model_64 = copy.deepcopy(model).double()

    set_seed(cfg.seed)
    # Marge `step` aux bords pour que x ± h reste dans le domaine
    x_star = (
        step + (1.0 - 2.0 * step) * torch.rand(n_points, 1, dtype=torch.float64)
    ).requires_grad_(True)
    y_star = (
        step + (1.0 - 2.0 * step) * torch.rand(n_points, 1, dtype=torch.float64)
    ).requires_grad_(True)
    t_star = (
        torch.rand(n_points, 1, dtype=torch.float64) * cfg.t_star_max
    ).requires_grad_(True)

    # --- Dérivées par autograd ------------------------------------------
    T_pred = model_64(x_star, y_star, t_star)
    T_x_autograd = gradient(T_pred, x_star, create_graph=True)
    T_xx_autograd = gradient(T_x_autograd, x_star, create_graph=False)

    # --- Dérivées par différences finies ---------------------------------
    with torch.no_grad():
        x_detached = x_star.detach()
        y_detached = y_star.detach()
        t_detached = t_star.detach()

        T_center = model_64(x_detached, y_detached, t_detached)
        T_plus = model_64(x_detached + step, y_detached, t_detached)
        T_minus = model_64(x_detached - step, y_detached, t_detached)

        T_x_finite = (T_plus - T_minus) / (2.0 * step)
        T_xx_finite = (T_plus - 2.0 * T_center + T_minus) / (step**2)

    # --- Comparaison relative --------------------------------------------
    # On relativise par l'échelle de la dérivée pour obtenir un nombre
    # sans dimension, comparable quelle que soit l'amplitude du réseau.
    first_scale = float(T_x_finite.abs().max()) + 1e-30
    second_scale = float(T_xx_finite.abs().max()) + 1e-30

    first_error = float((T_x_autograd.detach() - T_x_finite).abs().max()) / first_scale
    second_error = (
        float((T_xx_autograd.detach() - T_xx_finite).abs().max()) / second_scale
    )

    # Seuils volontairement LARGES au regard de la précision réellement
    # atteinte (≈1e-9 en tanh) : ils doivent couvrir aussi bien tanh que
    # SIREN, dont la référence FD est intrinsèquement moins précise.
    # Ils restent 4 à 6 ordres de grandeur plus serrés que ce que
    # produirait un vrai bug : une normalisation temporelle mal traversée
    # donnerait un facteur 2/t*_max = 20, soit une erreur relative ~1e1.
    assert first_error < 1e-5, (
        f"Autograd vs différences finies : ∂T*/∂x* diverge "
        f"(erreur relative {first_error:.3e})"
    )
    assert second_error < 1e-4, (
        f"Autograd vs différences finies : ∂²T*/∂x*² diverge "
        f"(erreur relative {second_error:.3e})"
    )

    print(
        f"  [OK] diff. finies : ∂T*/∂x* à {first_error:.2e}, "
        f"∂²T*/∂x*² à {second_error:.2e} (h={step}, float64)"
    )
    return first_error, second_error


# ==========================================================================
# 3c. Contre-exemple : pourquoi ReLU est bannie
# ==========================================================================

def demonstrate_relu_limitation(model: PINN, cfg: DataConfig) -> None:
    """
    Montre expérimentalement pourquoi une activation ReLU rend le PINN
    inopérant — argument à sortir en soutenance.

    ReLU est linéaire par morceaux : sa dérivée seconde est nulle presque
    partout. Un MLP à ReLU est donc une fonction affine par morceaux, dont
    le laplacien vaut EXACTEMENT 0 sur chaque morceau. Le résidu se
    réduirait alors à r = ∂T*/∂t*, et minimiser r reviendrait à demander
    un champ GELÉ dans le temps : la diffusion ne peut pas être apprise.

    On compare ici la norme du laplacien produit par le réseau tanh du
    projet à celle d'un réseau ReLU de même taille.
    """
    n_points = 512
    set_seed(cfg.seed)
    x_star = torch.rand(n_points, 1, requires_grad=True)
    y_star = torch.rand(n_points, 1, requires_grad=True)
    t_star = (torch.rand(n_points, 1) * cfg.t_star_max).requires_grad_(True)

    # Laplacien du réseau du projet (activation C²)
    tanh_derivatives = heat_derivatives(
        model, x_star, y_star, t_star, create_graph=False
    )
    tanh_laplacian = float(tanh_derivatives["laplacian"].abs().mean())

    # Même topologie, mais activations ReLU
    torch.manual_seed(cfg.seed)
    relu_network = nn.Sequential(
        nn.Linear(3, model.n_neurons),
        nn.ReLU(),
        nn.Linear(model.n_neurons, model.n_neurons),
        nn.ReLU(),
        nn.Linear(model.n_neurons, 1),
    )

    def relu_field(
        x: torch.Tensor, y: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return relu_network(torch.cat([x, y, t], dim=1))

    relu_derivatives = heat_derivatives(
        relu_field, x_star, y_star, t_star, create_graph=False
    )
    relu_laplacian = float(relu_derivatives["laplacian"].abs().mean())

    print(
        f"  [info] |Δ*T*| moyen — tanh : {tanh_laplacian:.3e}   "
        f"ReLU : {relu_laplacian:.3e}"
    )
    print(
        "         → le laplacien ReLU est numériquement nul : "
        "le résidu PDE ne contiendrait plus aucune information de diffusion."
    )


# ==========================================================================
# 4-5. Validation de la loss et du flux de gradient
# ==========================================================================

def validate_losses(
    model: PINN,
    ic_batch: CollocationBatch,
    bc_batch: CollocationBatch,
    res_batch: CollocationBatch,
    loss_cfg: LossConfig,
) -> LossTerms:
    """
    Évalue la loss multi-objectif sur le réseau NON ENTRAÎNÉ.

    Invariants contrôlés
    --------------------
    1. Chaque terme est un SCALAIRE (shape ()), positif, fini.
    2. `total` est bien la somme pondérée des trois termes.
    3. `total` est rattaché au graphe : sinon backward() serait impossible.

    Ordres de grandeur observés à l'initialisation (Xavier, 4×64, tanh) :
        L_ic  ~ 1e-1   le réseau sort un champ quelconque d'amplitude ~1,
                       alors que la cible vaut 0 presque partout et 1 dans
                       l'objet
        L_bc  ~ 1e-2   le plus petit des trois : la sortie initiale est
                       déjà proche de 0, donc les BC sont « satisfaites »
                       par accident
        L_res ~ 1e+1   LE TERME DOMINANT, de deux ordres de grandeur
                       au-dessus des autres

    POURQUOI L_res ÉCRASE-T-IL LES AUTRES ?
        L_ic et L_bc comparent des VALEURS (amplitude ~1). L_res compare
        des DÉRIVÉES SECONDES : dériver deux fois multiplie l'amplitude
        par le carré des fréquences spatiales du réseau, et la MSE élève
        encore au carré. Un champ initial parfaitement anodin produit donc
        un résidu de l'ordre de 10.

        Conséquence directe pour l'Étape 3 : avec des poids neutres
        (1, 1, 1), l'optimiseur consacrera d'abord presque tout son effort
        au résidu. Or le résidu seul admet la solution triviale T* ≡ 0.
        C'est précisément l'argument qui justifie la pondération dynamique
        au programme de l'Étape 3 — et ce déséquilibre se LIT sur la
        figure diagnostic.
    """
    terms = pinn_loss(model, ic_batch, bc_batch, res_batch, loss_cfg)

    for name, value in (
        ("L_ic", terms.ic),
        ("L_bc", terms.bc),
        ("L_res", terms.residual),
        ("L_total", terms.total),
    ):
        assert value.shape == (), f"{name} doit être un scalaire, shape {tuple(value.shape)}"
        assert torch.isfinite(value), f"{name} vaut NaN/inf"
        assert float(value) >= 0.0, f"{name} négatif ({float(value)}) — impossible pour une MSE"

    expected_total = (
        loss_cfg.w_ic * float(terms.ic.detach())
        + loss_cfg.w_bc * float(terms.bc.detach())
        + loss_cfg.w_res * float(terms.residual.detach())
    )
    assert abs(float(terms.total.detach()) - expected_total) < 1e-6 * max(expected_total, 1.0), (
        f"L_total ({float(terms.total.detach()):.6e}) ≠ somme pondérée ({expected_total:.6e})"
    )
    assert terms.total.grad_fn is not None, (
        "L_total n'est pas rattachée au graphe autograd : backward() échouerait"
    )

    print(
        f"  [OK] loss  : {terms}  "
        f"(poids {loss_cfg.w_ic}/{loss_cfg.w_bc}/{loss_cfg.w_res})"
    )
    return terms


def validate_gradient_flow(model: PINN, total_loss: torch.Tensor) -> None:
    """
    Vérifie que ∂L/∂θ atteint TOUS les paramètres du réseau.

    POURQUOI CE TEST EST LE DERNIER VERROU DE L'ÉTAPE 2
        Tout peut sembler correct — forward propre, résidu validé, loss
        calculée — et pourtant l'entraînement ne bougerait pas si le
        gradient n'atteignait pas les poids. Deux causes classiques :
          - un `.detach()` ou un `torch.no_grad()` égaré dans la chaîne ;
          - `create_graph=False` dans le calcul du résidu, qui coupe le
            lien entre la loss résidu et θ.
        Dans ce dernier cas L_res serait constante : le réseau
        apprendrait l'IC et les BC en ignorant totalement la physique.

    On contrôle donc que chaque paramètre reçoit un gradient non nul
    et fini après un unique backward().
    """
    model.zero_grad(set_to_none=True)
    total_loss.backward()

    n_parameters = 0
    n_without_gradient = 0
    max_gradient = 0.0

    for name, parameter in model.named_parameters():
        n_parameters += 1
        if parameter.grad is None:
            n_without_gradient += 1
            continue
        assert torch.isfinite(parameter.grad).all(), (
            f"Gradient non fini sur le paramètre '{name}'"
        )
        max_gradient = max(max_gradient, float(parameter.grad.abs().max()))

    assert n_without_gradient == 0, (
        f"{n_without_gradient}/{n_parameters} tenseurs de paramètres n'ont "
        "reçu AUCUN gradient — la chaîne autograd est rompue."
    )
    assert max_gradient > 0.0, (
        "Tous les gradients sont nuls : la loss ne dépend pas des poids."
    )

    print(
        f"  [OK] backward : {n_parameters}/{n_parameters} tenseurs de "
        f"paramètres reçoivent un gradient, |∂L/∂θ|_max = {max_gradient:.3e}"
    )


# ==========================================================================
# 6. Figure diagnostic
# ==========================================================================

def _evaluate_on_grid(
    model: PINN,
    cfg: DataConfig,
    t_star_value: float,
    resolution: int = 96,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Évalue T* et |r| sur une grille régulière à un instant t* donné.

    POURQUOI UNE GRILLE ALORS QUE LE PINN EST SANS MAILLAGE ?
        Uniquement pour l'AFFICHAGE : matplotlib a besoin d'un tableau
        2-D. Le réseau, lui, reste évaluable en n'importe quel point.
        C'est d'ailleurs un argument de vente du PINN : la résolution de
        la figure est un choix de post-traitement, pas une contrainte du
        modèle.

    Returns
    -------
    (X, Y, T_field, residual_field) : quatre np.ndarray de shape (R, R)
    """
    axis = np.linspace(0.0, 1.0, resolution)
    X, Y = np.meshgrid(axis, axis)

    x_flat = torch.tensor(
        X.ravel(), dtype=torch.float32, device=cfg.device
    ).unsqueeze(-1).requires_grad_(True)
    y_flat = torch.tensor(
        Y.ravel(), dtype=torch.float32, device=cfg.device
    ).unsqueeze(-1).requires_grad_(True)
    t_flat = torch.full_like(x_flat, t_star_value).requires_grad_(True)

    # create_graph=False : diagnostic pur, on n'entraîne rien ici
    derivatives = heat_derivatives(
        model, x_flat, y_flat, t_flat, create_graph=False
    )
    residual = derivatives["T_t"] - derivatives["laplacian"]

    T_field = derivatives["T"].detach().cpu().numpy().reshape(resolution, resolution)
    residual_field = residual.detach().cpu().numpy().reshape(resolution, resolution)
    return X, Y, T_field, residual_field


def plot_step2_diagnostics(
    model: PINN,
    cfg: DataConfig,
    model_cfg: ModelConfig,
    loss_cfg: LossConfig,
    terms: LossTerms,
    analytic_error: float,
    out_path: Path,
) -> None:
    """
    Figure diagnostic 2×2 de l'Étape 2.

    Layout
    ------
        (0, 0) Champ T* prédit à t* = 0 par le réseau NON ENTRAÎNÉ
               → bruit structuré : c'est le point de départ de l'Étape 3
        (0, 1) Carte du résidu |r| au même instant
               → montre que le résidu est calculable partout, sans maillage
        (1, 0) Les trois termes de la loss (échelle log)
               → visualise le déséquilibre initial entre L_ic, L_bc, L_res
        (1, 1) Cartouche : architecture, validations, équation

    Parameters
    ----------
    out_path : Path
        Chemin du PNG de sortie (ex. step2_pinn.png).
    """
    X, Y, T_field, residual_field = _evaluate_on_grid(model, cfg, t_star_value=0.0)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    figure.suptitle(
        "Étape 2 — Architecture du PINN & Dérivation Automatique\n"
        r"$r = \partial T^*/\partial t^* - "
        r"(\partial^2 T^*/\partial x^{*2} + \partial^2 T^*/\partial y^{*2})$"
        f"   |   réseau NON entraîné ({model.n_parameters:,} paramètres)",
        fontsize=13,
        fontweight="bold",
    )

    # ---- (0, 0) Champ T* prédit ---------------------------------------
    ax_field: Axes = axes[0, 0]
    field_map = ax_field.contourf(X, Y, T_field, levels=50, cmap="coolwarm")
    figure.colorbar(field_map, ax=ax_field, label=r"$T^*$ prédit")
    ax_field.set_xlabel(r"$x^*$")
    ax_field.set_ylabel(r"$y^*$")
    ax_field.set_aspect("equal")
    ax_field.set_title(
        r"$T^*_\theta(x^*, y^*, t^*=0)$ — avant entraînement"
    )

    # ---- (0, 1) Carte du résidu ---------------------------------------
    ax_residual: Axes = axes[0, 1]
    residual_map = ax_residual.contourf(
        X, Y, np.abs(residual_field), levels=50, cmap="magma"
    )
    figure.colorbar(residual_map, ax=ax_residual, label=r"$|r|$")
    ax_residual.set_xlabel(r"$x^*$")
    ax_residual.set_ylabel(r"$y^*$")
    ax_residual.set_aspect("equal")
    ax_residual.set_title("Résidu PDE calculé par autograd")

    # ---- (1, 0) Termes de la loss -------------------------------------
    ax_loss: Axes = axes[1, 0]
    loss_names = [
        r"$\mathcal{L}_{IC}$",
        r"$\mathcal{L}_{BC}$",
        r"$\mathcal{L}_{res}$",
        r"$\mathcal{L}_{total}$",
    ]
    loss_values = [
        float(terms.ic.detach()),
        float(terms.bc.detach()),
        float(terms.residual.detach()),
        float(terms.total.detach()),
    ]
    loss_colors = ["#55A868", "#F58518", "#4C72B0", "#3D405B"]
    bars = ax_loss.bar(
        loss_names, loss_values,
        color=loss_colors, edgecolor="black", linewidth=0.5,
    )
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("valeur (échelle log)")
    ax_loss.set_title(
        f"Termes de la loss à l'initialisation\n"
        f"$w$ = ({loss_cfg.w_ic}, {loss_cfg.w_bc}, {loss_cfg.w_res})"
    )
    for bar, value in zip(bars, loss_values):
        ax_loss.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2e}",
            ha="center", va="bottom", fontsize=8,
        )
    ax_loss.grid(True, axis="y", alpha=0.3)

    # ---- (1, 1) Cartouche ---------------------------------------------
    ax_text: Axes = axes[1, 1]
    ax_text.axis("off")
    hidden_topology = " → ".join(
        [str(model_cfg.n_neurons)] * model_cfg.n_hidden_layers
    )
    summary_text = (
        "Architecture T_θ\n"
        "─────────────────────────────\n"
        f"  3 → {hidden_topology} → 1\n"
        f"  activation : {model_cfg.activation} (classe C²)\n"
        f"  normalisation entrées : [-1, 1]\n"
        f"  paramètres : {model.n_parameters:,}\n"
        "\n"
        "Dérivation automatique\n"
        "─────────────────────────────\n"
        "  T_t   = grad(T, t*)\n"
        "  T_x   = grad(T, x*)\n"
        "  T_xx  = grad(T_x, x*)   ← create_graph=True\n"
        "  r     = T_t − (T_xx + T_yy)\n"
        "\n"
        "Validations\n"
        "─────────────────────────────\n"
        f"  solution analytique : r_rel = {analytic_error:.2e}\n"
        "  différences finies  : OK\n"
        "  flux du gradient    : OK\n"
        "\n"
        "Loss multi-objectif\n"
        "─────────────────────────────\n"
        f"  L = {loss_cfg.w_ic}·L_ic + {loss_cfg.w_bc}·L_bc "
        f"+ {loss_cfg.w_res}·L_res\n"
        f"  L_ic  = {float(terms.ic.detach()):.4e}\n"
        f"  L_bc  = {float(terms.bc.detach()):.4e}\n"
        f"  L_res = {float(terms.residual.detach()):.4e}\n"
        f"  L     = {float(terms.total.detach()):.4e}\n"
    )
    ax_text.text(
        0.05, 0.95, summary_text,
        transform=ax_text.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#F7F7F7", edgecolor="#CCCCCC"),
    )

    figure.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = Path(out_path)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"  [OK] Figure sauvegardée → {out_path.resolve()}")


# ==========================================================================
# Point d'entrée CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="PINN Thermal 2D — Étape 2 : architecture & autograd"
    )
    # Architecture
    parser.add_argument(
        "--layers", type=int, default=None,
        help="Nombre de couches cachées (défaut: ModelConfig=4)",
    )
    parser.add_argument(
        "--neurons", type=int, default=None,
        help="Neurones par couche cachée (défaut: ModelConfig=64)",
    )
    parser.add_argument(
        "--activation",
        choices=["tanh", "sin", "gelu", "softplus"],
        default=None,
        help="Activation C² du réseau (défaut: tanh)",
    )
    # Pondérations de la loss
    parser.add_argument(
        "--w-ic", type=float, default=1.0, help="Poids de L_ic (défaut: 1.0)",
    )
    parser.add_argument(
        "--w-bc", type=float, default=1.0, help="Poids de L_bc (défaut: 1.0)",
    )
    parser.add_argument(
        "--w-res", type=float, default=1.0, help="Poids de L_res (défaut: 1.0)",
    )
    # Échantillonnage (repris de l'Étape 1)
    parser.add_argument(
        "--method", choices=["sobol", "lhs", "uniform"], default="sobol",
        help="Stratégie d'échantillonnage (défaut: sobol)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine de reproductibilité",
    )
    parser.add_argument("--n-ic", type=int, default=None, help="Surcharge de N_ic")
    parser.add_argument("--n-bc", type=int, default=None, help="Surcharge de N_bc")
    parser.add_argument("--n-res", type=int, default=None, help="Surcharge de N_res")
    parser.add_argument(
        "--out", type=str, default="step2_pinn.png",
        help="Chemin de la figure PNG de sortie",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device PyTorch : 'cpu' ou 'cuda'",
    )
    return parser.parse_args()


def build_configs_from_args(
    args: argparse.Namespace,
) -> Tuple[DataConfig, ModelConfig, LossConfig]:
    """
    Construit les trois dataclasses de configuration depuis la CLI.

    Seuls les champs explicitement fournis (non-None) surchargent les
    valeurs par défaut : `python main_step2.py` sans aucun flag produit
    la configuration nominale du projet.
    """
    data_kwargs = {"seed": args.seed, "device": args.device}
    if args.n_ic is not None:
        data_kwargs["N_ic"] = args.n_ic
    if args.n_bc is not None:
        data_kwargs["N_bc"] = args.n_bc
    if args.n_res is not None:
        data_kwargs["N_res"] = args.n_res
    cfg = DataConfig(**data_kwargs)

    model_kwargs = {"seed": args.seed}
    if args.layers is not None:
        model_kwargs["n_hidden_layers"] = args.layers
    if args.neurons is not None:
        model_kwargs["n_neurons"] = args.neurons
    if args.activation is not None:
        model_kwargs["activation"] = args.activation
    model_cfg = ModelConfig(**model_kwargs)

    loss_cfg = LossConfig(w_ic=args.w_ic, w_bc=args.w_bc, w_res=args.w_res)
    return cfg, model_cfg, loss_cfg


def main() -> int:
    """
    Orchestration de l'Étape 2.

    Returns
    -------
    int
        Code retour shell (0 = succès).
    """
    args = parse_args()
    cfg, model_cfg, loss_cfg = build_configs_from_args(args)
    set_seed(cfg.seed)

    # ----- Bannière configuration --------------------------------------
    print(cfg.summary())
    print(model_cfg.summary())
    print(f"  {loss_cfg.summary()}")

    # ----- 1/6 Construction du réseau -----------------------------------
    print("\n[1/6] Construction du réseau T_θ…")
    model = PINN.from_config(cfg, model_cfg)
    print(model.summary())

    # ----- 2/6 Passe avant -----------------------------------------------
    print("[2/6] Validation de la passe avant…")
    validate_forward(model, cfg)

    # ----- 3/6 Validation de l'autograd ----------------------------------
    print("[3/6] Validation de la dérivation automatique…")
    analytic_error = validate_autograd_on_analytic(cfg)
    validate_autograd_vs_finite_differences(model, cfg)
    demonstrate_relu_limitation(model, cfg)

    # ----- 4/6 Génération des points (Étape 1) ---------------------------
    print("[4/6] Génération des points de collocation (Étape 1)…")
    ic_batch = sample_ic(cfg, method=args.method)
    bc_batch = sample_bc(cfg, method=args.method)
    res_batch = sample_residual(cfg, method=args.method)
    print(
        f"  [OK] IC={cfg.N_ic}, BC={cfg.N_bc}, RES={cfg.N_res} "
        f"(méthode {args.method})"
    )

    # ----- 5/6 Loss multi-objectif + flux du gradient ---------------------
    print("[5/6] Évaluation de la loss multi-objectif…")
    terms = validate_losses(model, ic_batch, bc_batch, res_batch, loss_cfg)
    validate_gradient_flow(model, terms.total)

    # ----- 6/6 Figure -----------------------------------------------------
    print("[6/6] Export de la figure diagnostic…")
    plot_step2_diagnostics(
        model, cfg, model_cfg, loss_cfg, terms, analytic_error,
        out_path=Path(args.out),
    )

    # ----- Récapitulatif --------------------------------------------------
    print("\n[confirm] Récapitulatif Étape 2 :")
    print(f"  Réseau        : {model.n_parameters:,} paramètres entraînables")
    print(f"  Autograd      : résidu relatif {analytic_error:.3e} sur solution exacte")
    print(f"  Loss initiale : {terms}")
    print(
        "  Termes pondérés : "
        f"w·L_ic={float(terms.weighted_ic.detach()):.4e}  "
        f"w·L_bc={float(terms.weighted_bc.detach()):.4e}  "
        f"w·L_res={float(terms.weighted_residual.detach()):.4e}"
    )

    print("\n✓ Étape 2 terminée avec succès.")
    print("  → Prochaine étape : entraînement hybride Adam → L-BFGS (Étape 3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
