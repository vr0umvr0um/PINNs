"""
==============================================================================
tests/test_losses.py — Tests unitaires de la loss multi-objectif (Étape 2)
==============================================================================

POURQUOI ces tests existent
---------------------------
Ils verrouillent l'équation (4) du sujet :

    L(θ) = w_ic · L_ic + w_bc · L_bc + w_res · L_res

Les invariants vérifiés sont ceux qui, s'ils cassaient, feraient
converger l'entraînement de l'Étape 3 vers une solution fausse SANS
message d'erreur :

    - chaque terme est un scalaire positif et fini
    - `total` est EXACTEMENT la somme pondérée (pas de terme oublié)
    - chaque terme lit la BONNE famille de points et la BONNE cible
    - la loss reste différentiable par rapport à θ de bout en bout

Lancer :
    pytest tests/test_losses.py -v
"""

from __future__ import annotations

import pytest
import torch

from config import DataConfig, LossConfig, ModelConfig
from src.losses import (
    LossTerms,
    loss_boundary,
    loss_initial,
    loss_residual,
    pinn_loss,
)
from src.models import PINN
from src.physics import analytic_solution
from src.sampling import sample_bc, sample_ic, sample_residual


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def cfg() -> DataConfig:
    """Configuration physique légère (budgets réduits)."""
    return DataConfig(N_ic=200, N_bc=200, N_res=400, seed=0, device="cpu")


@pytest.fixture
def model(cfg: DataConfig) -> PINN:
    """Réseau miniature non entraîné."""
    return PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=0))


@pytest.fixture
def batches(cfg: DataConfig):
    """Les trois familles de points de l'Étape 1."""
    return (
        sample_ic(cfg),
        sample_bc(cfg),
        sample_residual(cfg),
    )


def _zero_field(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Champ identiquement nul : T* ≡ 0.

    C'est la fameuse SOLUTION TRIVIALE du PINN. Elle vérifie parfaitement
    l'EDP (0 = 0) et les conditions aux limites (T*=0 sur les parois),
    mais rate complètement la condition initiale. On s'en sert pour
    vérifier que chaque terme mesure bien ce qu'il prétend mesurer.

    POURQUOI `0.0 * (x + y + t)` ET PAS `torch.zeros_like(x)` ?
        `zeros_like` crée un tenseur DÉTACHÉ du graphe : autograd
        refuserait de le dériver ("does not require grad"). L'écriture
        retenue vaut numériquement zéro tout en restant RATTACHÉE aux
        trois entrées, ce qui permet de calculer ∂T*/∂t* et le laplacien
        (tous deux nuls, comme attendu).
    """
    return 0.0 * (x + y + t)


# ==========================================================================
# Termes individuels
# ==========================================================================

class TestLossInitial:
    """L_ic = MSE( T_θ(x*, y*, 0), T*_cible )."""

    def test_is_scalar_and_finite(self, model: PINN, batches) -> None:
        """Une loss est un scalaire (shape ()), fini et positif."""
        ic_batch, _, _ = batches
        value = loss_initial(model, ic_batch)
        assert value.shape == ()
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0

    def test_zero_when_prediction_matches_target(self, batches) -> None:
        """Un « modèle » qui renvoie exactement la cible donne L_ic = 0."""
        ic_batch, _, _ = batches

        def perfect_field(x, y, t):
            return ic_batch["T_star"]

        assert float(loss_initial(perfect_field, ic_batch).detach()) == pytest.approx(0.0)

    def test_trivial_zero_field_fails_the_initial_condition(self, batches) -> None:
        """
        LE point pédagogique : T* ≡ 0 rate l'IC. La loss vaut alors la
        FRACTION de points situés dans l'objet chaud (chacun contribuant
        (0 − 1)² = 1, les autres 0).
        """
        ic_batch, _, _ = batches
        target = ic_batch["T_star"]
        hot_fraction = float((target > 0.5).float().mean())

        value = float(loss_initial(_zero_field, ic_batch).detach())
        assert value == pytest.approx(hot_fraction, abs=1e-6)
        assert value > 0.0, "T*≡0 devrait violer la condition initiale"

    def test_reads_the_target_from_the_batch(self, cfg: DataConfig) -> None:
        """
        La cible doit venir de ic_batch["T_star"], pas d'une constante
        codée en dur. On l'éprouve avec une IC uniforme à 1.
        """
        ic_batch = sample_ic(cfg, T_star_init=1.0)
        # T*≡0 face à une cible uniforme à 1 → MSE = 1 exactement
        assert float(loss_initial(_zero_field, ic_batch).detach()) == pytest.approx(1.0)


class TestLossBoundary:
    """L_bc = MSE( T_θ(paroi, t*), 0 )."""

    def test_is_scalar_and_finite(self, model: PINN, batches) -> None:
        """Scalaire, fini, positif."""
        _, bc_batch, _ = batches
        value = loss_boundary(model, bc_batch)
        assert value.shape == ()
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0

    def test_zero_field_satisfies_dirichlet(self, batches) -> None:
        """
        T* ≡ 0 satisfait EXACTEMENT les conditions de Dirichlet homogènes.
        C'est la moitié du piège de la solution triviale.
        """
        _, bc_batch, _ = batches
        assert float(loss_boundary(_zero_field, bc_batch).detach()) == pytest.approx(0.0)

    def test_analytic_solution_satisfies_dirichlet(self, batches) -> None:
        """
        La solution analytique s'annule sur les 4 parois : elle doit donc
        rendre L_bc numériquement nulle. Contrôle croisé entre physics.py
        et sampling.py (les points BC sont bien SUR le bord).
        """
        _, bc_batch, _ = batches
        assert float(loss_boundary(analytic_solution, bc_batch).detach()) < 1e-10

    def test_constant_offset_is_penalized(self, batches) -> None:
        """Un champ constant à 0.5 donne MSE = 0.25 sur les parois."""
        _, bc_batch, _ = batches

        def constant_field(x, y, t):
            return torch.full_like(x, 0.5)

        assert float(loss_boundary(constant_field, bc_batch).detach()) == pytest.approx(0.25)


class TestLossResidual:
    """L_res = MSE( ∂T*/∂t* − Δ*T*, 0 )."""

    def test_is_scalar_and_finite(self, model: PINN, batches) -> None:
        """Scalaire, fini, positif."""
        _, _, res_batch = batches
        value = loss_residual(model, res_batch)
        assert value.shape == ()
        assert torch.isfinite(value)
        assert float(value.detach()) >= 0.0

    def test_zero_field_satisfies_the_pde(self, batches) -> None:
        """
        T* ≡ 0 vérifie aussi l'EDP (0 = 0). Avec le test BC ci-dessus,
        cela démontre que SEUL L_ic empêche le réseau de s'effondrer sur
        la solution triviale — la justification directe du terme w_ic.
        """
        _, _, res_batch = batches
        assert float(loss_residual(_zero_field, res_batch).detach()) == pytest.approx(0.0)

    def test_analytic_solution_gives_near_zero_loss(self, cfg: DataConfig) -> None:
        """
        La solution exacte annule le résidu. On travaille en float64 :
        en float32 les dérivées secondes plafonneraient vers 1e-5.
        """
        res_batch = sample_residual(cfg)
        batch_64 = {
            key: tensor.detach().double().requires_grad_(True)
            for key, tensor in res_batch.items()
        }
        assert float(loss_residual(analytic_solution, batch_64).detach()) < 1e-18

    def test_non_solution_gives_expected_loss(self, cfg: DataConfig) -> None:
        """Pour T = x²+y²+t, le résidu vaut −3 partout → MSE = 9."""
        res_batch = sample_residual(cfg)
        batch_64 = {
            key: tensor.detach().double().requires_grad_(True)
            for key, tensor in res_batch.items()
        }

        def non_solution(x, y, t):
            return x**2 + y**2 + t

        assert float(loss_residual(non_solution, batch_64).detach()) == pytest.approx(9.0)


# ==========================================================================
# Loss totale
# ==========================================================================

class TestPINNLoss:
    """L = w_ic·L_ic + w_bc·L_bc + w_res·L_res (équation 4 du sujet)."""

    def test_returns_loss_terms(self, model: PINN, batches) -> None:
        """Le type de retour expose la décomposition, pas juste un scalaire."""
        ic_batch, bc_batch, res_batch = batches
        terms = pinn_loss(model, ic_batch, bc_batch, res_batch)
        assert isinstance(terms, LossTerms)

    def test_total_is_weighted_sum(self, model: PINN, batches) -> None:
        """
        INVARIANT CENTRAL : aucun terme ne doit être oublié ni compté
        deux fois dans la somme.
        """
        ic_batch, bc_batch, res_batch = batches
        loss_cfg = LossConfig(w_ic=3.0, w_bc=5.0, w_res=7.0)
        terms = pinn_loss(model, ic_batch, bc_batch, res_batch, loss_cfg)

        expected = (
            3.0 * float(terms.ic.detach())
            + 5.0 * float(terms.bc.detach())
            + 7.0 * float(terms.residual.detach())
        )
        assert float(terms.total.detach()) == pytest.approx(expected, rel=1e-6)

    def test_weights_scale_the_terms(self, model: PINN, batches) -> None:
        """Les termes pondérés valent bien w_i × terme brut."""
        ic_batch, bc_batch, res_batch = batches
        loss_cfg = LossConfig(w_ic=10.0, w_bc=2.0, w_res=0.5)
        terms = pinn_loss(model, ic_batch, bc_batch, res_batch, loss_cfg)

        assert float(terms.weighted_ic.detach()) == pytest.approx(10.0 * float(terms.ic.detach()))
        assert float(terms.weighted_bc.detach()) == pytest.approx(2.0 * float(terms.bc.detach()))
        assert float(terms.weighted_residual.detach()) == pytest.approx(
            0.5 * float(terms.residual.detach())
        )

    def test_raw_terms_are_independent_of_weights(self, model: PINN, batches) -> None:
        """
        Les termes BRUTS ne doivent pas dépendre des poids : c'est ce qui
        permet de comparer L_ic et L_res entre deux runs pondérés
        différemment (indispensable au diagnostic de l'Étape 3).
        """
        ic_batch, bc_batch, res_batch = batches
        neutral = pinn_loss(model, ic_batch, bc_batch, res_batch, LossConfig())
        weighted = pinn_loss(
            model, ic_batch, bc_batch, res_batch,
            LossConfig(w_ic=100.0, w_bc=0.01, w_res=42.0),
        )
        assert float(neutral.ic.detach()) == pytest.approx(float(weighted.ic.detach()))
        assert float(neutral.bc.detach()) == pytest.approx(float(weighted.bc.detach()))
        assert float(neutral.residual.detach()) == pytest.approx(float(weighted.residual.detach()))

    def test_zero_weights_cancel_a_term(self, model: PINN, batches) -> None:
        """Annuler w_res isole la partie supervisée (L_ic + L_bc)."""
        ic_batch, bc_batch, res_batch = batches
        terms = pinn_loss(
            model, ic_batch, bc_batch, res_batch,
            LossConfig(w_ic=1.0, w_bc=1.0, w_res=0.0),
        )
        expected = float(terms.ic.detach()) + float(terms.bc.detach())
        assert float(terms.total.detach()) == pytest.approx(expected, rel=1e-6)

    def test_gradient_reaches_every_parameter(self, model: PINN, batches) -> None:
        """
        LE verrou final de l'Étape 2 : un unique backward() doit fournir
        un gradient à TOUS les paramètres. Un seul tenseur sans gradient
        signalerait une chaîne autograd rompue, et l'entraînement de
        l'Étape 3 serait partiellement inopérant.
        """
        ic_batch, bc_batch, res_batch = batches
        terms = pinn_loss(model, ic_batch, bc_batch, res_batch)

        model.zero_grad(set_to_none=True)
        terms.total.backward()

        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Aucun gradient pour '{name}'"
            assert torch.isfinite(parameter.grad).all(), f"Gradient non fini : '{name}'"

        max_gradient = max(
            float(p.grad.abs().max()) for p in model.parameters()
        )
        assert max_gradient > 0.0

    def test_residual_term_contributes_to_the_gradient(
        self, model: PINN, batches
    ) -> None:
        """
        Le gradient obtenu avec w_res ≠ 0 doit DIFFÉRER de celui obtenu
        avec w_res = 0. Sinon le terme physique n'influencerait pas
        l'apprentissage — exactement le symptôme d'un create_graph=False
        oublié dans le calcul du résidu.
        """
        ic_batch, bc_batch, res_batch = batches
        first_parameter = next(iter(model.parameters()))

        model.zero_grad(set_to_none=True)
        pinn_loss(
            model, ic_batch, bc_batch, res_batch,
            LossConfig(w_ic=1.0, w_bc=1.0, w_res=0.0),
        ).total.backward()
        gradient_without_residual = first_parameter.grad.clone()

        model.zero_grad(set_to_none=True)
        pinn_loss(
            model, ic_batch, bc_batch, res_batch,
            LossConfig(w_ic=1.0, w_bc=1.0, w_res=1.0),
        ).total.backward()
        gradient_with_residual = first_parameter.grad.clone()

        assert not torch.allclose(
            gradient_without_residual, gradient_with_residual
        ), "Le terme résidu n'influence pas ∂L/∂θ"


class TestLossTerms:
    """Le conteneur de résultats (logging de l'Étape 3)."""

    def test_as_floats_returns_plain_floats(self, model: PINN, batches) -> None:
        """
        L'historique d'entraînement doit stocker des floats, pas des
        Tensors — sinon le graphe autograd resterait en mémoire à chaque
        epoch (fuite mémoire garantie sur plusieurs milliers d'itérations).
        """
        ic_batch, bc_batch, res_batch = batches
        values = pinn_loss(model, ic_batch, bc_batch, res_batch).as_floats()

        assert set(values) == {
            "total", "ic", "bc", "residual",
            "weighted_ic", "weighted_bc", "weighted_residual",
        }
        assert all(isinstance(value, float) for value in values.values())

    def test_str_is_readable(self, model: PINN, batches) -> None:
        """La ligne de log mentionne les trois termes."""
        ic_batch, bc_batch, res_batch = batches
        text = str(pinn_loss(model, ic_batch, bc_batch, res_batch))
        assert "L_ic" in text and "L_bc" in text and "L_res" in text
