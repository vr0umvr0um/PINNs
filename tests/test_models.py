"""
==============================================================================
tests/test_models.py — Tests unitaires de l'architecture (Étape 2)
==============================================================================

POURQUOI ces tests existent
---------------------------
Ils verrouillent le « contrat » de l'architecture, celui qu'on doit
pouvoir défendre en soutenance :

    - sortie de shape (N, 1), rattachée au graphe autograd
    - activation de classe C²  → ReLU explicitement REFUSÉE
    - normalisation affine des entrées vers [-1, 1], bornes issues de
      la physique (t*_max), et conservée dans le state_dict
    - ni BatchNorm ni Dropout (T_θ doit rester une fonction déterministe
      du seul point (x*, y*, t*))

Lancer :
    pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from config import DataConfig, ModelConfig
from src.models import PINN, InputNormalization, Sine, build_activation


# ==========================================================================
# Fixtures
# ==========================================================================

@pytest.fixture
def cfg() -> DataConfig:
    """Configuration physique légère pour les tests."""
    return DataConfig(N_ic=100, N_bc=100, N_res=200, seed=0, device="cpu")


@pytest.fixture
def model_cfg() -> ModelConfig:
    """
    Architecture MINIATURE pour les tests.

    2 couches × 8 neurones suffisent à vérifier les invariants
    structurels (shapes, autograd, normalisation) et rendent la suite
    quasi instantanée.
    """
    return ModelConfig(n_hidden_layers=2, n_neurons=8, seed=0)


@pytest.fixture
def model(cfg: DataConfig, model_cfg: ModelConfig) -> PINN:
    """Réseau miniature construit depuis les configs."""
    return PINN.from_config(cfg, model_cfg)


def _random_points(n_points: int, t_star_max: float):
    """Triplet (x*, y*, t*) aléatoire avec requires_grad=True."""
    x_star = torch.rand(n_points, 1, requires_grad=True)
    y_star = torch.rand(n_points, 1, requires_grad=True)
    t_star = (torch.rand(n_points, 1) * t_star_max).requires_grad_(True)
    return x_star, y_star, t_star


# ==========================================================================
# Activations
# ==========================================================================

class TestActivations:
    """L'activation doit être deux fois dérivable (résidu = dérivée 2nde)."""

    @pytest.mark.parametrize("name", ["tanh", "sin", "gelu", "softplus"])
    def test_supported_activations_are_modules(self, name: str) -> None:
        """Les quatre activations autorisées produisent bien un nn.Module."""
        assert isinstance(build_activation(name), nn.Module)

    def test_relu_is_rejected(self) -> None:
        """
        ReLU doit être REFUSÉE : sa dérivée seconde est nulle presque
        partout, donc le laplacien autograd serait identiquement 0.
        """
        with pytest.raises(ValueError, match="ReLU"):
            build_activation("relu")  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["tanh", "sin", "gelu", "softplus"])
    def test_second_derivative_is_non_trivial(self, name: str) -> None:
        """
        Invariant PHYSIQUE : chaque activation autorisée doit avoir une
        dérivée seconde NON nulle. C'est exactement ce que ReLU n'a pas,
        et sans quoi le PINN ne peut pas représenter la diffusion.
        """
        activation = build_activation(name)
        inputs = torch.linspace(-2.0, 2.0, 50).unsqueeze(-1).requires_grad_(True)

        outputs = activation(inputs)
        first = torch.autograd.grad(
            outputs, inputs, torch.ones_like(outputs), create_graph=True
        )[0]
        second = torch.autograd.grad(first, inputs, torch.ones_like(first))[0]

        assert float(second.abs().max()) > 1e-6, (
            f"L'activation {name} a une dérivée seconde numériquement nulle"
        )

    def test_sine_applies_omega(self) -> None:
        """Sine calcule bien sin(ω·x) et non sin(x)."""
        sine = Sine(omega=3.0)
        inputs = torch.tensor([[0.5]])
        assert torch.allclose(sine(inputs), torch.sin(torch.tensor([[1.5]])))


# ==========================================================================
# Normalisation des entrées
# ==========================================================================

class TestInputNormalization:
    """La normalisation doit envoyer exactement [lower, upper] sur [-1, 1]."""

    def test_maps_bounds_to_minus_one_and_one(self) -> None:
        """Les bornes basses → -1, les bornes hautes → +1."""
        normalizer = InputNormalization(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.1))

        lower_point = torch.tensor([[0.0, 0.0, 0.0]])
        upper_point = torch.tensor([[1.0, 1.0, 0.1]])

        assert torch.allclose(normalizer(lower_point), torch.full((1, 3), -1.0))
        assert torch.allclose(normalizer(upper_point), torch.full((1, 3), 1.0))

    def test_center_maps_to_zero(self) -> None:
        """Le centre du domaine tombe en 0, là où tanh est la plus sensible."""
        normalizer = InputNormalization(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.1))
        center = torch.tensor([[0.5, 0.5, 0.05]])
        assert torch.allclose(normalizer(center), torch.zeros(1, 3), atol=1e-7)

    def test_time_is_rescaled_like_space(self) -> None:
        """
        POINT CRITIQUE : t* ∈ [0, 0.1] doit couvrir la MÊME plage
        normalisée que x* ∈ [0, 1]. Sans cela, le temps serait 10× moins
        « visible » par le réseau qu'une coordonnée spatiale.
        """
        normalizer = InputNormalization(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.1))
        # Milieu de chaque domaine respectif
        point = torch.tensor([[0.25, 0.25, 0.025]])
        normalized = normalizer(point)
        assert normalized[0, 0] == pytest.approx(float(normalized[0, 2]), abs=1e-6)

    def test_rejects_degenerate_bounds(self) -> None:
        """Une plage nulle ou inversée doit lever une erreur explicite."""
        with pytest.raises(ValueError, match="Bornes de normalisation"):
            InputNormalization(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.0))

    def test_bounds_are_buffers_in_state_dict(self) -> None:
        """
        Les bornes doivent être des BUFFERS : sinon un modèle rechargé
        à l'Étape 3 perdrait sa normalisation et prédirait faux en
        silence.
        """
        normalizer = InputNormalization(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 0.1))
        state_dict = normalizer.state_dict()
        assert "lower" in state_dict
        assert "span" in state_dict


# ==========================================================================
# Le réseau PINN
# ==========================================================================

class TestPINNForward:
    """Contrat de la passe avant."""

    def test_output_shape(self, model: PINN, cfg: DataConfig) -> None:
        """3 tenseurs (N, 1) en entrée → 1 tenseur (N, 1) en sortie."""
        n_points = 37
        x_star, y_star, t_star = _random_points(n_points, cfg.t_star_max)
        assert tuple(model(x_star, y_star, t_star).shape) == (n_points, 1)

    def test_output_is_attached_to_graph(self, model: PINN, cfg: DataConfig) -> None:
        """Sans grad_fn, aucune dérivée — donc aucun résidu PDE possible."""
        x_star, y_star, t_star = _random_points(8, cfg.t_star_max)
        assert model(x_star, y_star, t_star).grad_fn is not None

    def test_output_is_finite(self, model: PINN, cfg: DataConfig) -> None:
        """Pas de NaN/inf dès l'initialisation Xavier."""
        x_star, y_star, t_star = _random_points(64, cfg.t_star_max)
        assert torch.isfinite(model(x_star, y_star, t_star)).all()

    def test_depends_on_every_input(self, model: PINN, cfg: DataConfig) -> None:
        """
        T_θ doit réellement dépendre des TROIS entrées. Une dérivée
        identiquement nulle par rapport à l'une d'elles signalerait une
        entrée écrasée par la normalisation ou oubliée dans la concaténation.
        """
        x_star, y_star, t_star = _random_points(64, cfg.t_star_max)
        T_pred = model(x_star, y_star, t_star)

        for name, variable in (("x*", x_star), ("y*", y_star), ("t*", t_star)):
            derivative = torch.autograd.grad(
                T_pred, variable, torch.ones_like(T_pred), retain_graph=True
            )[0]
            assert float(derivative.abs().max()) > 1e-8, (
                f"T_θ est insensible à {name}"
            )

    def test_predict_detaches_graph(self, model: PINN, cfg: DataConfig) -> None:
        """
        predict() sert à l'inférence (figures, Gradio) : la sortie ne doit
        PAS porter de graphe autograd, pour ne pas gaspiller de mémoire.
        """
        x_star, y_star, t_star = _random_points(8, cfg.t_star_max)
        assert model.predict(x_star, y_star, t_star).grad_fn is None

    def test_is_deterministic(self, model: PINN, cfg: DataConfig) -> None:
        """
        Deux évaluations du même point donnent la même valeur.
        Ce test échouerait si l'architecture contenait du Dropout —
        ce qui rendrait le résidu PDE aléatoire.
        """
        x_star, y_star, t_star = _random_points(16, cfg.t_star_max)
        first = model(x_star, y_star, t_star)
        second = model(x_star, y_star, t_star)
        assert torch.allclose(first, second)


class TestPINNArchitecture:
    """Contrat structurel du réseau."""

    def test_parameter_count(self) -> None:
        """
        Comptage explicite pour 2 couches cachées × 8 neurones :
            Linear(3→8)  : 3·8 + 8   = 32
            Linear(8→8)  : 8·8 + 8   = 72
            Linear(8→1)  : 8·1 + 1   = 9
            total                     = 113
        """
        model = PINN(n_hidden_layers=2, n_neurons=8, seed=0)
        assert model.n_parameters == 113

    def test_no_batchnorm_no_dropout(self, model: PINN) -> None:
        """
        Ni BatchNorm (rendrait T_θ dépendant des autres points du batch)
        ni Dropout (rendrait le résidu stochastique).
        """
        for module in model.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.Dropout)), (
                f"Module interdit dans un PINN : {type(module).__name__}"
            )

    def test_output_layer_is_linear(self, model: PINN) -> None:
        """
        La dernière couche doit être linéaire (pas de sigmoïde finale) :
        une saturation bloquerait les gradients pile sur les valeurs
        T*=0 et T*=1 de l'IC et des BC.
        """
        last_module = list(model.network.children())[-1]
        assert isinstance(last_module, nn.Linear)
        assert last_module.out_features == 1

    def test_biases_initialized_to_zero(self, model: PINN) -> None:
        """Xavier sur les poids, zéro sur les biais."""
        for module in model.network.modules():
            if isinstance(module, nn.Linear):
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))

    def test_rejects_zero_hidden_layers(self) -> None:
        """Un PINN sans couche cachée serait un modèle affine, donc de
        laplacien nul : refusé explicitement."""
        with pytest.raises(ValueError, match="n_hidden_layers"):
            PINN(n_hidden_layers=0)

    def test_same_seed_gives_same_weights(self) -> None:
        """Reproductibilité : deux réseaux de même graine sont identiques."""
        first = PINN(n_hidden_layers=2, n_neurons=8, seed=123)
        second = PINN(n_hidden_layers=2, n_neurons=8, seed=123)
        for p1, p2 in zip(first.parameters(), second.parameters()):
            assert torch.allclose(p1, p2)

    def test_different_seed_gives_different_weights(self) -> None:
        """…et deux graines différentes donnent des poids différents."""
        first = PINN(n_hidden_layers=2, n_neurons=8, seed=1)
        second = PINN(n_hidden_layers=2, n_neurons=8, seed=2)
        assert not torch.allclose(
            next(iter(first.parameters())), next(iter(second.parameters()))
        )


class TestPINNFromConfig:
    """Le constructeur depuis les dataclasses de configuration."""

    def test_time_bound_follows_physics(self, cfg: DataConfig) -> None:
        """
        INVARIANT CLÉ : la borne haute de normalisation du temps doit
        valoir cfg.t_star_max (≈ 0.1), PAS 1.0. C'est le même piège
        Fourier que celui verrouillé à l'Étape 1 pour l'échantillonnage.
        """
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=1, n_neurons=4))
        upper = (model.normalizer.lower + model.normalizer.span).flatten()
        assert float(upper[2]) == pytest.approx(cfg.t_star_max)
        assert float(upper[0]) == pytest.approx(1.0)
        assert float(upper[1]) == pytest.approx(1.0)

    def test_time_bound_tracks_config_changes(self) -> None:
        """
        Si l'on change la physique (alpha, t_max), la normalisation suit
        automatiquement — impossible d'oublier de la mettre à jour.
        """
        cfg = DataConfig(alpha=4e-5, t_max=5000.0)  # t*_max = 0.2
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=1, n_neurons=4))
        upper = (model.normalizer.lower + model.normalizer.span).flatten()
        assert float(upper[2]) == pytest.approx(0.2)

    def test_normalization_can_be_disabled(self, cfg: DataConfig) -> None:
        """L'option normalize_inputs=False retire bien le module."""
        model = PINN.from_config(
            cfg,
            ModelConfig(n_hidden_layers=1, n_neurons=4, normalize_inputs=False),
        )
        assert model.normalizer is None
