"""
==============================================================================
tests/test_physics.py — Tests unitaires de l'autograd & du résidu (Étape 2)
==============================================================================

POURQUOI ces tests sont les plus importants du projet
-----------------------------------------------------
Un bug d'autograd ne PLANTE pas : il renvoie silencieusement un résidu
faux. Le réseau converge alors très bien… vers la solution d'une autre
équation. Aucune inspection de la courbe de loss ne le révélerait.

La seule parade est de tester le résidu sur des fonctions dont on connaît
la réponse EXACTE :

    1. sin(πx)sin(πy)exp(−2π²t)  vérifie l'équation  →  r doit valoir 0
    2. x² + y² + t               ne la vérifie pas   →  r doit valoir −3

Le test 2 est indispensable : sans lui, une implémentation qui renverrait
bêtement zéro passerait le test 1 avec les honneurs.

Lancer :
    pytest tests/test_physics.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from config import DataConfig, ModelConfig
from src.models import PINN
from src.physics import (
    analytic_solution,
    gradient,
    heat_derivatives,
    pde_residual,
    relative_residual_error,
)


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def cfg() -> DataConfig:
    """Configuration physique légère."""
    return DataConfig(N_ic=100, N_bc=100, N_res=200, seed=0, device="cpu")


@pytest.fixture
def model(cfg: DataConfig) -> PINN:
    """Réseau miniature (les invariants d'autograd ne dépendent pas de la taille)."""
    return PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=0))


def _points_float64(n_points: int, t_star_max: float, seed: int = 0):
    """
    Points de collocation en float64.

    POURQUOI float64 ICI alors que le projet tourne en float32 ?
        Les dérivées secondes amplifient l'erreur d'arrondi. En float32
        (précision ~1e-7), le résidu d'une solution EXACTE plafonne vers
        1e-5 : correct, mais trop flou pour distinguer « exact » de
        « presque exact ». En float64 on attend ~1e-15, ce qui rend le
        test sans ambiguïté.
    """
    generator = torch.Generator().manual_seed(seed)
    x_star = torch.rand(n_points, 1, dtype=torch.float64, generator=generator)
    y_star = torch.rand(n_points, 1, dtype=torch.float64, generator=generator)
    t_star = torch.rand(n_points, 1, dtype=torch.float64, generator=generator)
    return (
        x_star.requires_grad_(True),
        y_star.requires_grad_(True),
        (t_star * t_star_max).requires_grad_(True),
    )


# ==========================================================================
# La brique `gradient`
# ==========================================================================

class TestGradient:
    """Vérifie le helper de dérivation avant de l'utiliser en cascade."""

    def test_derivative_of_square(self) -> None:
        """d(x²)/dx = 2x, pour chaque point du batch indépendamment."""
        x = torch.linspace(0.1, 2.0, 20, dtype=torch.float64).unsqueeze(-1)
        x.requires_grad_(True)
        derivative = gradient(x**2, x)
        assert torch.allclose(derivative, 2.0 * x, atol=1e-12)

    def test_second_derivative(self) -> None:
        """d²(x³)/dx² = 6x — nécessite create_graph=True au premier étage."""
        x = torch.linspace(0.1, 2.0, 20, dtype=torch.float64).unsqueeze(-1)
        x.requires_grad_(True)
        first = gradient(x**3, x, create_graph=True)
        second = gradient(first, x)
        assert torch.allclose(second, 6.0 * x, atol=1e-10)

    def test_batch_points_are_independent(self) -> None:
        """
        L'astuce `grad_outputs=ones` ne doit PAS mélanger les points :
        chaque ligne du résultat est la dérivée de son propre point.
        On le vérifie avec une fonction dont la dérivée varie fortement.
        """
        x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64, requires_grad=True)
        derivative = gradient(torch.exp(x), x)
        assert torch.allclose(derivative, torch.exp(x.detach()), atol=1e-12)

    def test_create_graph_false_detaches(self) -> None:
        """create_graph=False produit une dérivée non redérivable."""
        x = torch.ones(5, 1, requires_grad=True)
        derivative = gradient(x**2, x, create_graph=False)
        assert derivative.grad_fn is None


# ==========================================================================
# La solution analytique
# ==========================================================================

class TestAnalyticSolution:
    """La solution de référence doit être… une vraie solution."""

    def test_satisfies_heat_equation(self, cfg: DataConfig) -> None:
        """
        LE test central de l'Étape 2 : le résidu de la solution exacte
        doit tomber à la précision machine.
        """
        x_star, y_star, t_star = _points_float64(2048, cfg.t_star_max)
        error = relative_residual_error(analytic_solution, x_star, y_star, t_star)
        assert error < 1e-12, f"Résidu relatif {error:.3e} sur une solution EXACTE"

    @pytest.mark.parametrize("modes", [(1, 1), (2, 1), (2, 3)])
    def test_satisfies_equation_for_several_modes(
        self, cfg: DataConfig, modes
    ) -> None:
        """
        Tout mode propre (m, n) vérifie l'équation. Tester plusieurs modes
        exclut une coïncidence liée à la symétrie du mode fondamental.
        """
        x_star, y_star, t_star = _points_float64(1024, cfg.t_star_max)

        def field(x, y, t):
            return analytic_solution(x, y, t, modes=modes)

        error = relative_residual_error(field, x_star, y_star, t_star)
        assert error < 1e-12

    def test_time_derivative_matches_theory(self, cfg: DataConfig) -> None:
        """∂T*/∂t* = −(m²+n²)π² · T*, ici −2π²·T* pour le mode (1, 1)."""
        x_star, y_star, t_star = _points_float64(512, cfg.t_star_max)
        derivatives = heat_derivatives(
            analytic_solution, x_star, y_star, t_star, create_graph=False
        )
        expected = -2.0 * (math.pi**2) * derivatives["T"]
        assert torch.allclose(derivatives["T_t"], expected, atol=1e-10)

    def test_second_derivatives_match_theory(self, cfg: DataConfig) -> None:
        """∂²T*/∂x*² = −π²·T* et ∂²T*/∂y*² = −π²·T* pour le mode (1, 1)."""
        x_star, y_star, t_star = _points_float64(512, cfg.t_star_max)
        derivatives = heat_derivatives(
            analytic_solution, x_star, y_star, t_star, create_graph=False
        )
        expected = -(math.pi**2) * derivatives["T"]
        assert torch.allclose(derivatives["T_xx"], expected, atol=1e-10)
        assert torch.allclose(derivatives["T_yy"], expected, atol=1e-10)

    def test_vanishes_on_walls(self) -> None:
        """
        La solution de référence respecte aussi les BC de Dirichlet
        homogènes : elle s'annule sur les 4 parois du carré unité.
        """
        wall_points = torch.tensor(
            [[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0]], dtype=torch.float64
        )
        x_star = wall_points[:, 0:1]
        y_star = wall_points[:, 1:2]
        t_star = torch.full_like(x_star, 0.05)
        values = analytic_solution(x_star, y_star, t_star)
        assert torch.allclose(values, torch.zeros_like(values), atol=1e-14)


# ==========================================================================
# Le résidu PDE
# ==========================================================================

class TestPDEResidual:
    """Contrat du résidu r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)."""

    def test_non_solution_gives_expected_value(self, cfg: DataConfig) -> None:
        """
        CONTRE-ÉPREUVE INDISPENSABLE.
        Pour T = x² + y² + t :  ∂T/∂t = 1,  ΔT = 2 + 2 = 4,  donc r = −3.
        Sans ce test, un pde_residual qui renverrait toujours 0 passerait
        tous les tests précédents.
        """
        x_star, y_star, t_star = _points_float64(256, cfg.t_star_max)

        def non_solution(x, y, t):
            return x**2 + y**2 + t

        residual = pde_residual(
            non_solution, x_star, y_star, t_star, create_graph=False
        )
        assert torch.allclose(
            residual, torch.full_like(residual, -3.0), atol=1e-10
        )

    def test_shape_matches_input(self, model: PINN, cfg: DataConfig) -> None:
        """Le résidu a une valeur par point de collocation : shape (N, 1)."""
        n_points = 43
        x_star = torch.rand(n_points, 1, requires_grad=True)
        y_star = torch.rand(n_points, 1, requires_grad=True)
        t_star = (torch.rand(n_points, 1) * cfg.t_star_max).requires_grad_(True)
        assert tuple(pde_residual(model, x_star, y_star, t_star).shape) == (n_points, 1)

    def test_laplacian_is_sum_of_second_derivatives(
        self, model: PINN, cfg: DataConfig
    ) -> None:
        """Δ*T* = ∂²T*/∂x*² + ∂²T*/∂y*², par définition."""
        x_star = torch.rand(32, 1, requires_grad=True)
        y_star = torch.rand(32, 1, requires_grad=True)
        t_star = (torch.rand(32, 1) * cfg.t_star_max).requires_grad_(True)
        derivatives = heat_derivatives(model, x_star, y_star, t_star)
        assert torch.allclose(
            derivatives["laplacian"], derivatives["T_xx"] + derivatives["T_yy"]
        )

    def test_requires_grad_is_enforced(self, model: PINN, cfg: DataConfig) -> None:
        """
        Un tenseur sans requires_grad doit produire un message EXPLICITE,
        pas un obscur RuntimeError d'autograd.
        """
        x_star = torch.rand(8, 1)  # requires_grad=False volontairement
        y_star = torch.rand(8, 1, requires_grad=True)
        t_star = torch.rand(8, 1, requires_grad=True)
        with pytest.raises(RuntimeError, match="requires_grad"):
            pde_residual(model, x_star, y_star, t_star)

    def test_residual_is_differentiable_wrt_parameters(
        self, model: PINN, cfg: DataConfig
    ) -> None:
        """
        Avec create_graph=True, la loss résidu doit pouvoir remonter
        jusqu'aux poids θ. C'est ce qui rend l'entraînement possible :
        sans cela, le réseau ignorerait totalement la physique.
        """
        x_star = torch.rand(32, 1, requires_grad=True)
        y_star = torch.rand(32, 1, requires_grad=True)
        t_star = (torch.rand(32, 1) * cfg.t_star_max).requires_grad_(True)

        residual = pde_residual(model, x_star, y_star, t_star, create_graph=True)
        (residual**2).mean().backward()

        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients, "Le résidu ne produit aucun gradient sur θ"
        assert max(float(g.abs().max()) for g in gradients) > 0.0

    def test_residual_alone_cannot_constrain_the_output_bias(
        self, model: PINN, cfg: DataConfig
    ) -> None:
        """
        RÉSULTAT PHYSIQUE, PAS UN BUG — et un excellent argument de
        soutenance.

        Le biais de la couche de SORTIE ajoute une constante à T*. Or
        l'opérateur de la chaleur annule les constantes :

            ∂(T + c)/∂t* − Δ*(T + c)  =  ∂T/∂t* − Δ*T

        Ce biais est donc dans le NOYAU de l'opérateur : le résidu n'en
        dépend pas du tout, et `backward()` ne lui attribue aucun
        gradient. Conséquence directe : l'EDP seule ne détermine la
        solution qu'à une constante additive près. Ce sont l'IC et les BC
        qui fixent cette constante — d'où la nécessité des trois termes
        de la loss, démontrée ici numériquement.
        """
        x_star = torch.rand(32, 1, requires_grad=True)
        y_star = torch.rand(32, 1, requires_grad=True)
        t_star = (torch.rand(32, 1) * cfg.t_star_max).requires_grad_(True)

        model.zero_grad(set_to_none=True)
        residual = pde_residual(model, x_star, y_star, t_star, create_graph=True)
        (residual**2).mean().backward()

        output_layer = list(model.network.children())[-1]
        assert output_layer.bias.grad is None, (
            "Le biais de sortie ne devrait recevoir aucun gradient du résidu seul"
        )
        # …alors que le POIDS de la même couche, lui, en reçoit bien un.
        assert output_layer.weight.grad is not None
        assert float(output_layer.weight.grad.abs().max()) > 0.0

    def test_relu_network_has_zero_laplacian(self) -> None:
        """
        Justification EXPÉRIMENTALE du refus de ReLU dans l'architecture.

        Un MLP à ReLU est affine par morceaux : son laplacien vaut
        exactement 0 sur chaque morceau. Le résidu se réduirait à
        r = ∂T*/∂t*, et le minimiser reviendrait à exiger un champ gelé —
        la diffusion serait inapprenable.
        """
        torch.manual_seed(0)
        relu_network = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )

        def relu_field(x, y, t):
            return relu_network(torch.cat([x, y, t], dim=1))

        x_star = torch.rand(128, 1, requires_grad=True)
        y_star = torch.rand(128, 1, requires_grad=True)
        t_star = (torch.rand(128, 1) * 0.1).requires_grad_(True)

        derivatives = heat_derivatives(
            relu_field, x_star, y_star, t_star, create_graph=False
        )
        assert float(derivatives["laplacian"].abs().max()) == 0.0

    def test_tanh_network_has_non_zero_laplacian(
        self, model: PINN, cfg: DataConfig
    ) -> None:
        """Le pendant positif : le réseau du projet produit un vrai laplacien."""
        x_star = torch.rand(128, 1, requires_grad=True)
        y_star = torch.rand(128, 1, requires_grad=True)
        t_star = (torch.rand(128, 1) * cfg.t_star_max).requires_grad_(True)
        derivatives = heat_derivatives(
            model, x_star, y_star, t_star, create_graph=False
        )
        assert float(derivatives["laplacian"].abs().max()) > 1e-6


# ==========================================================================
# Cohérence avec le reste du projet
# ==========================================================================

class TestAutogradVersusFiniteDifferences:
    """
    Confronte l'autograd du VRAI réseau à des différences finies.

    Ce test valide la chaîne complète (concaténation → normalisation →
    couches). Il détecterait notamment une normalisation temporelle non
    prise en compte, qui donnerait un ∂T*/∂t* faux d'un facteur 2/t*_max.
    """

    def test_first_derivative(self, cfg: DataConfig) -> None:
        """∂T*/∂x* autograd ≈ [T(x+h) − T(x−h)] / 2h."""
        model = PINN.from_config(
            cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=0)
        ).double()
        step = 1e-4

        x_star = torch.full((32, 1), 0.5, dtype=torch.float64, requires_grad=True)
        y_star = torch.full((32, 1), 0.3, dtype=torch.float64, requires_grad=True)
        t_star = torch.full(
            (32, 1), 0.5 * cfg.t_star_max, dtype=torch.float64, requires_grad=True
        )

        autograd_derivative = gradient(
            model(x_star, y_star, t_star), x_star, create_graph=False
        )
        with torch.no_grad():
            x_detached = x_star.detach()
            finite_derivative = (
                model(x_detached + step, y_star.detach(), t_star.detach())
                - model(x_detached - step, y_star.detach(), t_star.detach())
            ) / (2.0 * step)

        assert torch.allclose(autograd_derivative, finite_derivative, atol=1e-7)

    def test_time_derivative_uses_fourier_scale(self, cfg: DataConfig) -> None:
        """
        PIÈGE CIBLÉ : ∂T*/∂t* doit être exprimée dans la variable de
        Fourier t*, PAS dans la variable normalisée interne. Un écart
        d'un facteur 2/t*_max = 20 trahirait une normalisation mal
        traversée par autograd.
        """
        model = PINN.from_config(
            cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=0)
        ).double()
        step = 1e-6

        x_star = torch.full((16, 1), 0.5, dtype=torch.float64, requires_grad=True)
        y_star = torch.full((16, 1), 0.5, dtype=torch.float64, requires_grad=True)
        t_star = torch.full(
            (16, 1), 0.5 * cfg.t_star_max, dtype=torch.float64, requires_grad=True
        )

        autograd_derivative = gradient(
            model(x_star, y_star, t_star), t_star, create_graph=False
        )
        with torch.no_grad():
            t_detached = t_star.detach()
            finite_derivative = (
                model(x_star.detach(), y_star.detach(), t_detached + step)
                - model(x_star.detach(), y_star.detach(), t_detached - step)
            ) / (2.0 * step)

        assert torch.allclose(autograd_derivative, finite_derivative, rtol=1e-5)
