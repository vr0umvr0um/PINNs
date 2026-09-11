"""
Tests unitaires — Étape 1 : adimensionnement & échantillonnage.

Vérifie :
    - dimensions (shapes) des tenseurs IC / BC / résidu
    - t* = 0 pour la condition initiale
    - T* = 0 pour les conditions aux limites Dirichlet
    - requires_grad=True sur les coordonnées (x*, y*, t*)
    - appartenance au domaine [0, 1]
    - répartition équilibrée des points BC sur les 4 parois
    - helpers d'adimensionnement de DataConfig
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import DataConfig
from src.sampling import sample_bc, sample_ic, sample_residual, sample_all
from src.utils import (
    latin_hypercube,
    mask_boundary,
    mask_interior,
    sobol_sample,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> DataConfig:
    """Config légère pour tests rapides."""
    return DataConfig(
        N_ic=200,
        N_bc=200,
        N_res=500,
        seed=0,
        device="cpu",
    )


@pytest.fixture
def cfg_full() -> DataConfig:
    """Config aux budgets nominaux du cahier des charges."""
    return DataConfig(seed=42, device="cpu")


# ---------------------------------------------------------------------------
# DataConfig / adimensionnement
# ---------------------------------------------------------------------------

class TestDataConfig:
    def test_default_values(self):
        c = DataConfig()
        assert c.Lx == 1.0
        assert c.Ly == 1.0
        assert c.alpha == 2e-5
        assert c.T_amb == 20.0
        assert c.T_obj == 80.0
        assert c.N_ic == 2000
        assert c.N_bc == 2000
        assert c.N_res == 20000

    def test_delta_T(self):
        c = DataConfig()
        assert c.delta_T == 60.0

    def test_to_T_star_bounds(self):
        c = DataConfig()
        assert c.to_T_star(c.T_amb) == pytest.approx(0.0)
        assert c.to_T_star(c.T_obj) == pytest.approx(1.0)
        assert c.from_T_star(0.0) == pytest.approx(c.T_amb)
        assert c.from_T_star(1.0) == pytest.approx(c.T_obj)

    def test_to_x_y_star(self):
        c = DataConfig(Lx=1.0, Ly=1.0)
        assert c.to_x_star(0.5) == pytest.approx(0.5)
        assert c.to_y_star(1.0) == pytest.approx(1.0)

    def test_t_ref_and_fourier(self):
        c = DataConfig(Lx=1.0, Ly=1.0, alpha=2e-5, t_max=5000.0)
        assert c.t_ref == pytest.approx(1.0 / 2e-5)
        assert c.t_star_max == pytest.approx(5000.0 * 2e-5 / 1.0)

    def test_summary_str(self):
        text = DataConfig().summary()
        assert "DataConfig" in text
        assert "α=" in text or "alpha" in text.lower() or "α" in text


# ---------------------------------------------------------------------------
# Utils — Sobol / LHS / masques
# ---------------------------------------------------------------------------

class TestUtils:
    def test_sobol_shape_and_bounds(self):
        pts = sobol_sample(128, dim=3, seed=1)
        assert pts.shape == (128, 3)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_lhs_shape_and_bounds(self):
        pts = latin_hypercube(64, dim=2, seed=1)
        assert pts.shape == (64, 2)
        assert pts.min() >= 0.0
        assert pts.max() <= 1.0

    def test_sobol_with_bounds(self):
        bounds = np.array([[0.0, 2.0], [-1.0, 1.0]])
        pts = sobol_sample(32, dim=2, bounds=bounds, seed=0)
        assert pts[:, 0].min() >= 0.0 and pts[:, 0].max() <= 2.0
        assert pts[:, 1].min() >= -1.0 and pts[:, 1].max() <= 1.0

    def test_mask_interior_and_boundary(self):
        x = np.array([0.0, 0.5, 1.0, 0.5, 0.2])
        y = np.array([0.5, 0.0, 0.5, 0.5, 0.3])
        interior = mask_interior(x, y)
        boundary = mask_boundary(x, y)
        assert interior.tolist() == [False, False, False, True, True]
        assert boundary.tolist() == [True, True, True, False, False]


# ---------------------------------------------------------------------------
# sample_ic
# ---------------------------------------------------------------------------

class TestSampleIC:
    def test_shapes(self, cfg: DataConfig):
        ic = sample_ic(cfg)
        n = cfg.N_ic
        assert ic["x_star"].shape == (n, 1)
        assert ic["y_star"].shape == (n, 1)
        assert ic["t_star"].shape == (n, 1)
        assert ic["T_star"].shape == (n, 1)

    def test_t_star_is_zero(self, cfg: DataConfig):
        """Invariant clé : t* = 0 pour toute la condition initiale."""
        ic = sample_ic(cfg)
        assert torch.allclose(ic["t_star"], torch.zeros_like(ic["t_star"]))

    def test_T_star_default_zero(self, cfg: DataConfig):
        ic = sample_ic(cfg)
        assert torch.allclose(ic["T_star"], torch.zeros_like(ic["T_star"]))

    def test_T_star_custom(self, cfg: DataConfig):
        ic = sample_ic(cfg, T_star_init=0.5)
        assert torch.allclose(ic["T_star"], torch.full_like(ic["T_star"], 0.5))

    def test_requires_grad_on_coordinates(self, cfg: DataConfig):
        ic = sample_ic(cfg)
        assert ic["x_star"].requires_grad is True
        assert ic["y_star"].requires_grad is True
        assert ic["t_star"].requires_grad is True
        assert ic["T_star"].requires_grad is False

    def test_spatial_domain(self, cfg: DataConfig):
        ic = sample_ic(cfg)
        x = ic["x_star"].detach()
        y = ic["y_star"].detach()
        assert torch.all(x >= 0) and torch.all(x <= 1)
        assert torch.all(y >= 0) and torch.all(y <= 1)

    def test_full_budget_shapes(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        assert ic["x_star"].shape == (2000, 1)
        assert ic["t_star"].shape == (2000, 1)

    @pytest.mark.parametrize("method", ["sobol", "lhs", "uniform"])
    def test_methods(self, cfg: DataConfig, method: str):
        ic = sample_ic(cfg, method=method)  # type: ignore[arg-type]
        assert ic["x_star"].shape[0] == cfg.N_ic
        assert torch.allclose(ic["t_star"], torch.zeros_like(ic["t_star"]))


# ---------------------------------------------------------------------------
# sample_bc
# ---------------------------------------------------------------------------

class TestSampleBC:
    def test_shapes(self, cfg: DataConfig):
        bc = sample_bc(cfg)
        n = cfg.N_bc
        assert bc["x_star"].shape == (n, 1)
        assert bc["y_star"].shape == (n, 1)
        assert bc["t_star"].shape == (n, 1)
        assert bc["T_star"].shape == (n, 1)
        assert bc["wall"].shape == (n, 1)

    def test_T_star_is_zero(self, cfg: DataConfig):
        """Invariant clé : T* = 0 sur les conditions aux limites (Dirichlet ambiante)."""
        bc = sample_bc(cfg)
        assert torch.allclose(bc["T_star"], torch.zeros_like(bc["T_star"]))

    def test_requires_grad_on_coordinates(self, cfg: DataConfig):
        bc = sample_bc(cfg)
        assert bc["x_star"].requires_grad is True
        assert bc["y_star"].requires_grad is True
        assert bc["t_star"].requires_grad is True
        assert bc["T_star"].requires_grad is False

    def test_points_on_boundary(self, cfg: DataConfig):
        bc = sample_bc(cfg)
        x = bc["x_star"].detach().cpu().numpy().ravel()
        y = bc["y_star"].detach().cpu().numpy().ravel()
        on_edge = (
            np.isclose(x, 0.0)
            | np.isclose(x, 1.0)
            | np.isclose(y, 0.0)
            | np.isclose(y, 1.0)
        )
        assert on_edge.all()

    def test_wall_distribution_balanced(self, cfg: DataConfig):
        bc = sample_bc(cfg)
        walls = bc["wall"].detach().cpu().numpy().ravel()
        counts = [int((walls == w).sum()) for w in range(4)]
        # Chaque paroi reçoit floor(N/4) ou ceil(N/4)
        assert sum(counts) == cfg.N_bc
        assert max(counts) - min(counts) <= 1

    def test_time_in_unit_interval(self, cfg: DataConfig):
        bc = sample_bc(cfg)
        t = bc["t_star"].detach()
        assert torch.all(t >= 0) and torch.all(t <= 1)

    def test_full_budget_shapes(self, cfg_full: DataConfig):
        bc = sample_bc(cfg_full)
        assert bc["T_star"].shape == (2000, 1)
        assert torch.allclose(bc["T_star"], torch.zeros(2000, 1))


# ---------------------------------------------------------------------------
# sample_residual
# ---------------------------------------------------------------------------

class TestSampleResidual:
    def test_shapes(self, cfg: DataConfig):
        res = sample_residual(cfg)
        n = cfg.N_res
        assert res["x_star"].shape == (n, 1)
        assert res["y_star"].shape == (n, 1)
        assert res["t_star"].shape == (n, 1)
        assert "T_star" not in res  # résidu : pas de cible Dirichlet

    def test_requires_grad_on_coordinates(self, cfg: DataConfig):
        res = sample_residual(cfg)
        assert res["x_star"].requires_grad is True
        assert res["y_star"].requires_grad is True
        assert res["t_star"].requires_grad is True

    def test_strictly_interior(self, cfg: DataConfig):
        res = sample_residual(cfg)
        for key in ("x_star", "y_star", "t_star"):
            v = res[key].detach()
            assert torch.all(v > 0.0) and torch.all(v < 1.0)

    def test_full_budget_shapes(self, cfg_full: DataConfig):
        res = sample_residual(cfg_full)
        assert res["x_star"].shape == (20000, 1)


# ---------------------------------------------------------------------------
# sample_all + dtypes
# ---------------------------------------------------------------------------

class TestSampleAll:
    def test_keys_and_dtypes(self, cfg: DataConfig):
        bundles = sample_all(cfg)
        assert set(bundles.keys()) == {"ic", "bc", "res"}
        for bundle in bundles.values():
            for key, tensor in bundle.items():
                if key == "wall":
                    assert tensor.dtype == torch.int64
                else:
                    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Invariants croisés cahier des charges
# ---------------------------------------------------------------------------

class TestSpecInvariants:
    """Reprise explicite des contraintes du brief Étape 1."""

    def test_ic_t_star_zero_full(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        assert torch.all(ic["t_star"] == 0)

    def test_bc_T_star_zero_full(self, cfg_full: DataConfig):
        bc = sample_bc(cfg_full)
        assert torch.all(bc["T_star"] == 0)

    def test_nominal_shapes(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        bc = sample_bc(cfg_full)
        res = sample_residual(cfg_full)
        assert ic["x_star"].shape == (2000, 1)
        assert bc["x_star"].shape == (2000, 1)
        assert res["x_star"].shape == (20000, 1)
