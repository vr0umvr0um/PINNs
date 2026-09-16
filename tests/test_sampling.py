"""
==============================================================================
tests/test_sampling.py — Tests unitaires de l'Étape 1
==============================================================================

POURQUOI ces tests existent
---------------------------
Ils verrouillent le "contrat" mathématique de l'Étape 1, celui qu'on
doit pouvoir réciter en soutenance :

    IC  : t* = 0,  T* ∈ {0, 1} (objet chaud peints), shape (N_ic, 1)
    BC  : T* = 0,  t* ∈ [0, t*_max ≈ 0.1], points sur les 4 parois
    RES : (x*, y*) ∈ (0, 1)², t* ∈ (0, t*_max), requires_grad=True

Si un refactoring casse l'un de ces invariants, `pytest` le détecte
immédiatement — bien avant la démo live.

Lancer :
    pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import DataConfig
from src.sampling import sample_all, sample_bc, sample_ic, sample_residual
from src.utils import (
    latin_hypercube,
    mask_boundary,
    mask_hot_object,
    mask_interior,
    sobol_sample,
)


# ==========================================================================
# Fixtures — configurations de test réutilisables
# ==========================================================================

@pytest.fixture
def cfg() -> DataConfig:
    """
    Config LÉGÈRE pour les tests rapides.

    POURQUOI réduire N_ic / N_bc / N_res ?
        Les tests d'invariants (shapes, bornes, requires_grad) ne
        dépendent PAS du budget. 200/200/500 points suffisent et
        rendent la suite ~10× plus rapide que les budgets nominaux.
    """
    return DataConfig(
        N_ic=200,
        N_bc=200,
        N_res=500,
        seed=0,
        device="cpu",
    )


@pytest.fixture
def cfg_full() -> DataConfig:
    """
    Config aux budgets NOMINAUX du cahier des charges.

    Utilisée uniquement pour les tests "full budget" qui vérifient
    explicitement N_ic=2000, N_bc=2000, N_res=20000.
    """
    return DataConfig(seed=42, device="cpu")


# ==========================================================================
# DataConfig / adimensionnement
# ==========================================================================

class TestDataConfig:
    """Vérifie les constantes et les conversions d'adimensionnement."""

    def test_default_values(self) -> None:
        """Les valeurs par défaut du brief doivent être respectées."""
        config = DataConfig()
        assert config.Lx == 1.0
        assert config.Ly == 1.0
        assert config.alpha == 2e-5
        assert config.T_amb == 20.0
        assert config.T_obj == 80.0
        assert config.N_ic == 2000
        assert config.N_bc == 2000
        assert config.N_res == 20000

    def test_delta_T(self) -> None:
        """ΔT = T_obj - T_amb = 60°C (dénominateur de T*)."""
        config = DataConfig()
        assert config.delta_T == 60.0

    def test_to_T_star_bounds(self) -> None:
        """
        Bornes de T* : T_amb → 0, T_obj → 1.
        C'est la définition même de l'adimensionnement en température.
        """
        config = DataConfig()
        assert config.to_T_star(config.T_amb) == pytest.approx(0.0)
        assert config.to_T_star(config.T_obj) == pytest.approx(1.0)
        assert config.from_T_star(0.0) == pytest.approx(config.T_amb)
        assert config.from_T_star(1.0) == pytest.approx(config.T_obj)

    def test_to_x_y_star(self) -> None:
        """x* = x/Lx, y* = y/Ly (ici L=1 → identité)."""
        config = DataConfig(Lx=1.0, Ly=1.0)
        assert config.to_x_star(0.5) == pytest.approx(0.5)
        assert config.to_y_star(1.0) == pytest.approx(1.0)

    def test_t_ref_and_fourier(self) -> None:
        """
        t_ref = L²/α = 50 000 s
        t*_max = t_max / t_ref = 0.1
        """
        config = DataConfig(Lx=1.0, Ly=1.0, alpha=2e-5, t_max=5000.0)
        assert config.t_ref == pytest.approx(1.0 / 2e-5)
        assert config.t_star_max == pytest.approx(0.1)
        assert config.t_star_max == pytest.approx(5000.0 * 2e-5 / 1.0)

    def test_t_star_range_is_fourier(self) -> None:
        """
        POINT CRITIQUE : t* vit dans [0, t*_max ≈ 0.1], PAS [0, 1].
        """
        config = DataConfig()
        assert config.t_star_range == (0.0, config.t_star_max)
        assert config.t_star_range[1] == pytest.approx(0.1)
        assert config.t_star_range[1] < 1.0

    def test_to_t_star_fourier(self) -> None:
        """to_t_star utilise t_ref (Fourier), pas t_max."""
        config = DataConfig()
        assert config.to_t_star(0.0) == pytest.approx(0.0)
        assert config.to_t_star(config.t_max) == pytest.approx(config.t_star_max)
        assert config.from_t_star(config.t_star_max) == pytest.approx(config.t_max)

    def test_point_in_object_disk(self) -> None:
        """Appartenance au disque chaud (intérieur strict / extérieur)."""
        config = DataConfig(
            obj_shape="disk", obj_cx=0.5, obj_cy=0.5, obj_radius=0.15
        )
        assert config.point_in_object(0.5, 0.5) is True  # centre
        assert config.point_in_object(0.5 + 0.14, 0.5) is True  # dans le disque
        assert config.point_in_object(0.5 + 0.16, 0.5) is False  # hors disque
        assert config.point_in_object(0.9, 0.9) is False

    def test_summary_str(self) -> None:
        """summary() doit produire un texte lisible contenant DataConfig."""
        text = DataConfig().summary()
        assert "DataConfig" in text
        assert "α=" in text or "alpha" in text.lower() or "α" in text


# ==========================================================================
# Utils — Sobol / LHS / masques
# ==========================================================================

class TestUtils:
    """Tests des helpers d'échantillonnage et de masquage géométrique."""

    def test_sobol_shape_and_bounds(self) -> None:
        """Sobol renvoie (n, d) dans [0, 1]^d."""
        points = sobol_sample(n_points=128, n_dimensions=3, seed=1)
        assert points.shape == (128, 3)
        assert points.min() >= 0.0
        assert points.max() <= 1.0

    def test_lhs_shape_and_bounds(self) -> None:
        """LHS renvoie (n, d) dans [0, 1]^d."""
        points = latin_hypercube(n_points=64, n_dimensions=2, seed=1)
        assert points.shape == (64, 2)
        assert points.min() >= 0.0
        assert points.max() <= 1.0

    def test_sobol_with_bounds(self) -> None:
        """Sobol + bounds scale correctement chaque dimension."""
        bounds = np.array([[0.0, 2.0], [-1.0, 1.0]])
        points = sobol_sample(
            n_points=32, n_dimensions=2, bounds=bounds, seed=0
        )
        assert points[:, 0].min() >= 0.0 and points[:, 0].max() <= 2.0
        assert points[:, 1].min() >= -1.0 and points[:, 1].max() <= 1.0

    def test_mask_interior_and_boundary(self) -> None:
        """
        Points de test :
            (0.0, 0.5) bord   | (0.5, 0.0) bord
            (1.0, 0.5) bord   | (0.5, 0.5) intérieur
            (0.2, 0.3) intérieur
        """
        x_star = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        y_star = np.array([0.5, 0.0, 0.5, 0.5, 0.3])
        interior = mask_interior(x_star, y_star)
        boundary = mask_boundary(x_star, y_star)
        assert interior.tolist() == [False, False, False, True, True]
        assert boundary.tolist() == [True, True, True, False, False]

    def test_mask_hot_object_disk(self) -> None:
        """Disque r=0.15 centré : centre et point proche = dedans."""
        x_star = np.array([0.5, 0.5, 0.9, 0.5 + 0.1])
        y_star = np.array([0.5, 0.9, 0.9, 0.5])
        mask = mask_hot_object(
            x_star, y_star,
            center_x=0.5, center_y=0.5, radius=0.15, shape="disk",
        )
        assert mask.tolist() == [True, False, False, True]

    def test_mask_hot_object_square(self) -> None:
        """Carré demi-côté 0.15 : norme infinie."""
        x_star = np.array([0.5, 0.5 + 0.1, 0.5 + 0.2])
        y_star = np.array([0.5, 0.5 + 0.1, 0.5])
        mask = mask_hot_object(
            x_star, y_star,
            center_x=0.5, center_y=0.5, radius=0.15, shape="square",
        )
        assert mask.tolist() == [True, True, False]


# ==========================================================================
# sample_ic
# ==========================================================================

class TestSampleIC:
    """Tests de la condition initiale (t*=0, objet chaud)."""

    def test_shapes(self, cfg: DataConfig) -> None:
        """Tous les tenseurs IC ont la shape (N_ic, 1)."""
        ic_batch = sample_ic(cfg)
        n_ic = cfg.N_ic
        assert ic_batch["x_star"].shape == (n_ic, 1)
        assert ic_batch["y_star"].shape == (n_ic, 1)
        assert ic_batch["t_star"].shape == (n_ic, 1)
        assert ic_batch["T_star"].shape == (n_ic, 1)

    def test_t_star_is_zero(self, cfg: DataConfig) -> None:
        """Invariant clé : t* = 0 pour toute la condition initiale."""
        ic_batch = sample_ic(cfg)
        assert torch.allclose(
            ic_batch["t_star"], torch.zeros_like(ic_batch["t_star"])
        )

    def test_T_star_has_hot_and_cold(self, cfg: DataConfig) -> None:
        """
        L'IC doit contenir T*=1 (objet) ET T*=0 (extérieur).
        → T* ∈ [0.00, 1.00], valeurs uniquement dans {0, 1}.
        """
        ic_batch = sample_ic(cfg)
        T_star = ic_batch["T_star"].detach()
        assert float(T_star.min()) == pytest.approx(0.0)
        assert float(T_star.max()) == pytest.approx(1.0)
        unique_values = set(torch.unique(T_star).tolist())
        assert unique_values <= {0.0, 1.0}
        assert (T_star == 1.0).any() and (T_star == 0.0).any()

    def test_T_star_matches_object_mask(self, cfg: DataConfig) -> None:
        """T* peint par le masque doit coïncider point par point."""
        ic_batch = sample_ic(cfg)
        x_star = ic_batch["x_star"].detach().cpu().numpy().ravel()
        y_star = ic_batch["y_star"].detach().cpu().numpy().ravel()
        T_star = ic_batch["T_star"].detach().cpu().numpy().ravel()

        inside_object = mask_hot_object(
            x_star, y_star,
            center_x=cfg.obj_cx, center_y=cfg.obj_cy,
            radius=cfg.obj_radius, shape=cfg.obj_shape,
        )
        assert np.allclose(T_star[inside_object], 1.0)
        assert np.allclose(T_star[~inside_object], 0.0)

    def test_T_star_uniform_override(self, cfg: DataConfig) -> None:
        """Override optionnel : IC uniforme (mode test uniquement)."""
        ic_batch = sample_ic(cfg, T_star_init=0.5)
        assert torch.allclose(
            ic_batch["T_star"], torch.full_like(ic_batch["T_star"], 0.5)
        )

    def test_requires_grad_on_coordinates(self, cfg: DataConfig) -> None:
        """
        x*, y*, t* : requires_grad=True  (entrées du réseau / autograd)
        T*         : requires_grad=False (cible constante)
        """
        ic_batch = sample_ic(cfg)
        assert ic_batch["x_star"].requires_grad is True
        assert ic_batch["y_star"].requires_grad is True
        assert ic_batch["t_star"].requires_grad is True
        assert ic_batch["T_star"].requires_grad is False

    def test_spatial_domain(self, cfg: DataConfig) -> None:
        """(x*, y*) ∈ [0, 1]²."""
        ic_batch = sample_ic(cfg)
        x_star = ic_batch["x_star"].detach()
        y_star = ic_batch["y_star"].detach()
        assert torch.all(x_star >= 0) and torch.all(x_star <= 1)
        assert torch.all(y_star >= 0) and torch.all(y_star <= 1)

    def test_full_budget_shapes_and_T_range(self, cfg_full: DataConfig) -> None:
        """Budgets nominaux + T* ∈ [0, 1]."""
        ic_batch = sample_ic(cfg_full)
        assert ic_batch["x_star"].shape == (2000, 1)
        assert ic_batch["t_star"].shape == (2000, 1)
        T_star = ic_batch["T_star"].detach()
        assert float(T_star.min()) == pytest.approx(0.0)
        assert float(T_star.max()) == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ["sobol", "lhs", "uniform"])
    def test_methods(self, cfg: DataConfig, method: str) -> None:
        """Les 3 méthodes produisent un batch IC valide."""
        ic_batch = sample_ic(cfg, method=method)  # type: ignore[arg-type]
        assert ic_batch["x_star"].shape[0] == cfg.N_ic
        assert torch.allclose(
            ic_batch["t_star"], torch.zeros_like(ic_batch["t_star"])
        )
        assert float(ic_batch["T_star"].max()) == pytest.approx(1.0)

    def test_square_object(self, cfg: DataConfig) -> None:
        """L'objet carré produit aussi T* ∈ {0, 1}."""
        cfg.obj_shape = "square"
        ic_batch = sample_ic(cfg)
        assert float(ic_batch["T_star"].max()) == pytest.approx(1.0)
        assert float(ic_batch["T_star"].min()) == pytest.approx(0.0)


# ==========================================================================
# sample_bc
# ==========================================================================

class TestSampleBC:
    """Tests des conditions aux limites Dirichlet."""

    def test_shapes(self, cfg: DataConfig) -> None:
        """Tous les tenseurs BC ont la shape (N_bc, 1)."""
        bc_batch = sample_bc(cfg)
        n_bc = cfg.N_bc
        assert bc_batch["x_star"].shape == (n_bc, 1)
        assert bc_batch["y_star"].shape == (n_bc, 1)
        assert bc_batch["t_star"].shape == (n_bc, 1)
        assert bc_batch["T_star"].shape == (n_bc, 1)
        assert bc_batch["wall"].shape == (n_bc, 1)

    def test_T_star_is_zero(self, cfg: DataConfig) -> None:
        """Invariant clé : T* = 0 sur les parois (Dirichlet ambiante)."""
        bc_batch = sample_bc(cfg)
        assert torch.allclose(
            bc_batch["T_star"], torch.zeros_like(bc_batch["T_star"])
        )

    def test_requires_grad_on_coordinates(self, cfg: DataConfig) -> None:
        """requires_grad=True sur les coordonnées, False sur T*."""
        bc_batch = sample_bc(cfg)
        assert bc_batch["x_star"].requires_grad is True
        assert bc_batch["y_star"].requires_grad is True
        assert bc_batch["t_star"].requires_grad is True
        assert bc_batch["T_star"].requires_grad is False

    def test_points_on_boundary(self, cfg: DataConfig) -> None:
        """Chaque point BC est sur au moins une paroi (x*=0/1 ou y*=0/1)."""
        bc_batch = sample_bc(cfg)
        x_star = bc_batch["x_star"].detach().cpu().numpy().ravel()
        y_star = bc_batch["y_star"].detach().cpu().numpy().ravel()
        on_edge = (
            np.isclose(x_star, 0.0)
            | np.isclose(x_star, 1.0)
            | np.isclose(y_star, 0.0)
            | np.isclose(y_star, 1.0)
        )
        assert on_edge.all()

    def test_wall_distribution_balanced(self, cfg: DataConfig) -> None:
        """Les 4 parois reçoivent floor(N/4) ou ceil(N/4) points."""
        bc_batch = sample_bc(cfg)
        wall_ids = bc_batch["wall"].detach().cpu().numpy().ravel()
        counts = [int((wall_ids == wall_id).sum()) for wall_id in range(4)]
        assert sum(counts) == cfg.N_bc
        assert max(counts) - min(counts) <= 1

    def test_time_in_fourier_interval(self, cfg: DataConfig) -> None:
        """
        POINT CRITIQUE : t* ∈ [0, t*_max ≈ 0.1], PAS [0, 1].
        """
        bc_batch = sample_bc(cfg)
        t_star = bc_batch["t_star"].detach()
        t_star_max = cfg.t_star_max

        assert t_star_max == pytest.approx(0.1)
        assert torch.all(t_star >= 0)
        assert torch.all(t_star <= t_star_max + 1e-9)
        # Doit explorer le haut de l'intervalle (pas collé à 0)
        assert float(t_star.max()) > 0.5 * t_star_max
        # Ne doit PAS atteindre ~1.0 (ancien bug)
        assert float(t_star.max()) < 0.5

    def test_full_budget_shapes(self, cfg_full: DataConfig) -> None:
        """Budgets nominaux + t* ≤ t*_max."""
        bc_batch = sample_bc(cfg_full)
        assert bc_batch["T_star"].shape == (2000, 1)
        assert torch.allclose(bc_batch["T_star"], torch.zeros(2000, 1))
        t_star = bc_batch["t_star"].detach()
        assert float(t_star.max()) <= cfg_full.t_star_max + 1e-9


# ==========================================================================
# sample_residual
# ==========================================================================

class TestSampleResidual:
    """Tests des points de collocation du résidu PDE."""

    def test_shapes(self, cfg: DataConfig) -> None:
        """Shapes (N_res, 1) ; pas de clé T_star (pas de cible Dirichlet)."""
        res_batch = sample_residual(cfg)
        n_res = cfg.N_res
        assert res_batch["x_star"].shape == (n_res, 1)
        assert res_batch["y_star"].shape == (n_res, 1)
        assert res_batch["t_star"].shape == (n_res, 1)
        assert "T_star" not in res_batch

    def test_requires_grad_on_coordinates(self, cfg: DataConfig) -> None:
        """
        requires_grad=True OBLIGATOIRE : c'est sur ces tenseurs que
        l'Étape 2 appellera torch.autograd.grad pour construire le laplacien.
        """
        res_batch = sample_residual(cfg)
        assert res_batch["x_star"].requires_grad is True
        assert res_batch["y_star"].requires_grad is True
        assert res_batch["t_star"].requires_grad is True

    def test_spatial_strictly_interior(self, cfg: DataConfig) -> None:
        """(x*, y*) ∈ (0, 1)² — pas de point collé au bord."""
        res_batch = sample_residual(cfg)
        for key in ("x_star", "y_star"):
            values = res_batch[key].detach()
            assert torch.all(values > 0.0) and torch.all(values < 1.0)

    def test_time_in_fourier_interval(self, cfg: DataConfig) -> None:
        """
        POINT CRITIQUE : t* ∈ (0, t*_max ≈ 0.1), PAS (0, 1).
        """
        res_batch = sample_residual(cfg)
        t_star = res_batch["t_star"].detach()
        t_star_max = cfg.t_star_max

        assert t_star_max == pytest.approx(0.1)
        assert torch.all(t_star > 0.0)
        assert torch.all(t_star < t_star_max)
        assert float(t_star.max()) > 0.5 * t_star_max
        assert float(t_star.max()) < 0.5  # largement sous 1.0

    def test_full_budget_shapes(self, cfg_full: DataConfig) -> None:
        """Budget nominal N_res=20000 + t* ∈ (0, t*_max)."""
        res_batch = sample_residual(cfg_full)
        assert res_batch["x_star"].shape == (20000, 1)
        t_star = res_batch["t_star"].detach()
        assert float(t_star.max()) < cfg_full.t_star_max
        assert float(t_star.min()) > 0.0


# ==========================================================================
# sample_all + dtypes
# ==========================================================================

class TestSampleAll:
    """Test du raccourci sample_all et des dtypes."""

    def test_keys_and_dtypes(self, cfg: DataConfig) -> None:
        """
        sample_all renvoie les 3 familles.
        float32 partout sauf wall (int64).
        """
        bundles = sample_all(cfg)
        assert set(bundles.keys()) == {"ic", "bc", "res"}
        for batch in bundles.values():
            for key, tensor in batch.items():
                if key == "wall":
                    assert tensor.dtype == torch.int64
                else:
                    assert tensor.dtype == torch.float32


# ==========================================================================
# Invariants croisés du cahier des charges
# ==========================================================================

class TestSpecInvariants:
    """
    Reprise EXPLICITE des contraintes du brief Étape 1.

    Ces tests sont les plus "soutenance-friendly" : chacun correspond
    à une phrase du cahier des charges qu'on doit pouvoir justifier.
    """

    def test_ic_t_star_zero_full(self, cfg_full: DataConfig) -> None:
        """Brief : t* = 0 pour l'IC."""
        ic_batch = sample_ic(cfg_full)
        assert torch.all(ic_batch["t_star"] == 0)

    def test_ic_T_star_range_full(self, cfg_full: DataConfig) -> None:
        """Brief : T* ∈ [0, 1] sur l'IC (objet chaud)."""
        ic_batch = sample_ic(cfg_full)
        T_star = ic_batch["T_star"].detach()
        assert float(T_star.min()) == pytest.approx(0.0)
        assert float(T_star.max()) == pytest.approx(1.0)

    def test_bc_T_star_zero_full(self, cfg_full: DataConfig) -> None:
        """Brief : T* = 0 pour les BC."""
        bc_batch = sample_bc(cfg_full)
        assert torch.all(bc_batch["T_star"] == 0)

    def test_time_domain_is_fourier_not_unit(self, cfg_full: DataConfig) -> None:
        """
        Brief corrigé : t* ∈ [0, t*_max ≈ 0.1] pour BC et résidu.
        On rejette explicitement l'ancien bug t* ∈ [0, 1].
        """
        bc_batch = sample_bc(cfg_full)
        res_batch = sample_residual(cfg_full)
        t_star_max = cfg_full.t_star_max

        assert t_star_max == pytest.approx(0.1)
        assert float(bc_batch["t_star"].detach().max()) <= t_star_max + 1e-9
        assert float(res_batch["t_star"].detach().max()) < t_star_max
        # Rejet explicite de l'ancien domaine [0, 1]
        assert float(bc_batch["t_star"].detach().max()) < 0.5
        assert float(res_batch["t_star"].detach().max()) < 0.5

    def test_nominal_shapes(self, cfg_full: DataConfig) -> None:
        """Brief : N_ic=2000, N_bc=2000, N_res=20000."""
        ic_batch = sample_ic(cfg_full)
        bc_batch = sample_bc(cfg_full)
        res_batch = sample_residual(cfg_full)
        assert ic_batch["x_star"].shape == (2000, 1)
        assert bc_batch["x_star"].shape == (2000, 1)
        assert res_batch["x_star"].shape == (20000, 1)
