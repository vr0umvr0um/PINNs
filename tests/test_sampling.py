"""
Tests unitaires — Étape 1 : adimensionnement & échantillonnage.

Vérifie :
    - dimensions (shapes) des tenseurs IC / BC / résidu
    - t* = 0 pour la condition initiale
    - T* ∈ {0, 1} pour l'IC (objet chaud → 1, extérieur → 0)
    - T* = 0 pour les conditions aux limites Dirichlet
    - t* ∈ [0, t*_max] pour BC et résidu (t*_max ≈ 0.1, PAS [0, 1])
    - requires_grad=True sur les coordonnées (x*, y*, t*)
    - appartenance au domaine spatial [0, 1]
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
    mask_hot_object,
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
        assert c.t_star_max == pytest.approx(0.1)
        assert c.t_star_max == pytest.approx(5000.0 * 2e-5 / 1.0)

    def test_t_star_range_is_fourier(self):
        """t* doit vivre dans [0, t*_max≈0.1], pas [0, 1]."""
        c = DataConfig()
        assert c.t_star_range == (0.0, c.t_star_max)
        assert c.t_star_range[1] == pytest.approx(0.1)
        assert c.t_star_range[1] < 1.0

    def test_to_t_star_fourier(self):
        c = DataConfig()
        assert c.to_t_star(0.0) == pytest.approx(0.0)
        assert c.to_t_star(c.t_max) == pytest.approx(c.t_star_max)
        assert c.from_t_star(c.t_star_max) == pytest.approx(c.t_max)

    def test_point_in_object_disk(self):
        c = DataConfig(obj_shape="disk", obj_cx=0.5, obj_cy=0.5, obj_radius=0.15)
        assert c.point_in_object(0.5, 0.5) is True
        assert c.point_in_object(0.5 + 0.14, 0.5) is True  # strictly inside
        assert c.point_in_object(0.5 + 0.16, 0.5) is False  # strictly outside
        assert c.point_in_object(0.9, 0.9) is False

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

    def test_mask_hot_object_disk(self):
        x = np.array([0.5, 0.5, 0.9, 0.5 + 0.1])
        y = np.array([0.5, 0.9, 0.9, 0.5])
        m = mask_hot_object(x, y, cx=0.5, cy=0.5, radius=0.15, shape="disk")
        assert m.tolist() == [True, False, False, True]

    def test_mask_hot_object_square(self):
        x = np.array([0.5, 0.5 + 0.1, 0.5 + 0.2])
        y = np.array([0.5, 0.5 + 0.1, 0.5])
        m = mask_hot_object(x, y, cx=0.5, cy=0.5, radius=0.15, shape="square")
        assert m.tolist() == [True, True, False]


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

    def test_T_star_has_hot_and_cold(self, cfg: DataConfig):
        """
        BUG FIX : l'IC doit contenir T*=1 dans l'objet chaud et T*=0 dehors.
        → T* ∈ [0.00, 1.00]
        """
        ic = sample_ic(cfg)
        T = ic["T_star"].detach()
        assert float(T.min()) == pytest.approx(0.0)
        assert float(T.max()) == pytest.approx(1.0)
        # Uniquement les deux valeurs {0, 1}
        unique = torch.unique(T)
        assert set(unique.tolist()) <= {0.0, 1.0}
        assert (T == 1.0).any() and (T == 0.0).any()

    def test_T_star_matches_object_mask(self, cfg: DataConfig):
        ic = sample_ic(cfg)
        x = ic["x_star"].detach().cpu().numpy().ravel()
        y = ic["y_star"].detach().cpu().numpy().ravel()
        T = ic["T_star"].detach().cpu().numpy().ravel()
        inside = mask_hot_object(
            x, y,
            cx=cfg.obj_cx, cy=cfg.obj_cy,
            radius=cfg.obj_radius, shape=cfg.obj_shape,
        )
        assert np.allclose(T[inside], 1.0)
        assert np.allclose(T[~inside], 0.0)

    def test_T_star_uniform_override(self, cfg: DataConfig):
        """Override optionnel pour tests : IC uniforme."""
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

    def test_full_budget_shapes_and_T_range(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        assert ic["x_star"].shape == (2000, 1)
        assert ic["t_star"].shape == (2000, 1)
        T = ic["T_star"].detach()
        assert float(T.min()) == pytest.approx(0.0)
        assert float(T.max()) == pytest.approx(1.0)

    @pytest.mark.parametrize("method", ["sobol", "lhs", "uniform"])
    def test_methods(self, cfg: DataConfig, method: str):
        ic = sample_ic(cfg, method=method)  # type: ignore[arg-type]
        assert ic["x_star"].shape[0] == cfg.N_ic
        assert torch.allclose(ic["t_star"], torch.zeros_like(ic["t_star"]))
        assert float(ic["T_star"].max()) == pytest.approx(1.0)

    def test_square_object(self, cfg: DataConfig):
        cfg.obj_shape = "square"
        ic = sample_ic(cfg)
        assert float(ic["T_star"].max()) == pytest.approx(1.0)
        assert float(ic["T_star"].min()) == pytest.approx(0.0)


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

    def test_time_in_fourier_interval(self, cfg: DataConfig):
        """
        BUG FIX : t* ∈ [0, t*_max] avec t*_max ≈ 0.1, PAS [0, 1].
        """
        bc = sample_bc(cfg)
        t = bc["t_star"].detach()
        t_max_star = cfg.t_star_max
        assert t_max_star == pytest.approx(0.1)
        assert torch.all(t >= 0)
        assert torch.all(t <= t_max_star + 1e-9)
        # Doit effectivement explorer le haut de l'intervalle (pas collé à 0)
        assert float(t.max()) > 0.5 * t_max_star
        # Ne doit PAS atteindre 1.0
        assert float(t.max()) < 0.5  # largement sous 1.0

    def test_full_budget_shapes(self, cfg_full: DataConfig):
        bc = sample_bc(cfg_full)
        assert bc["T_star"].shape == (2000, 1)
        assert torch.allclose(bc["T_star"], torch.zeros(2000, 1))
        t = bc["t_star"].detach()
        assert float(t.max()) <= cfg_full.t_star_max + 1e-9


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

    def test_spatial_strictly_interior(self, cfg: DataConfig):
        res = sample_residual(cfg)
        for key in ("x_star", "y_star"):
            v = res[key].detach()
            assert torch.all(v > 0.0) and torch.all(v < 1.0)

    def test_time_in_fourier_interval(self, cfg: DataConfig):
        """
        BUG FIX : t* ∈ (0, t*_max) avec t*_max ≈ 0.1, PAS (0, 1).
        """
        res = sample_residual(cfg)
        t = res["t_star"].detach()
        t_max_star = cfg.t_star_max
        assert t_max_star == pytest.approx(0.1)
        assert torch.all(t > 0.0)
        assert torch.all(t < t_max_star)
        assert float(t.max()) > 0.5 * t_max_star
        assert float(t.max()) < 0.5  # largement sous 1.0

    def test_full_budget_shapes(self, cfg_full: DataConfig):
        res = sample_residual(cfg_full)
        assert res["x_star"].shape == (20000, 1)
        t = res["t_star"].detach()
        assert float(t.max()) < cfg_full.t_star_max
        assert float(t.min()) > 0.0


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
    """Reprise explicite des contraintes du brief Étape 1 (+ bugfixes)."""

    def test_ic_t_star_zero_full(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        assert torch.all(ic["t_star"] == 0)

    def test_ic_T_star_range_full(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        T = ic["T_star"].detach()
        assert float(T.min()) == pytest.approx(0.0)
        assert float(T.max()) == pytest.approx(1.0)

    def test_bc_T_star_zero_full(self, cfg_full: DataConfig):
        bc = sample_bc(cfg_full)
        assert torch.all(bc["T_star"] == 0)

    def test_time_domain_is_fourier_not_unit(self, cfg_full: DataConfig):
        bc = sample_bc(cfg_full)
        res = sample_residual(cfg_full)
        t_max = cfg_full.t_star_max
        assert t_max == pytest.approx(0.1)
        assert float(bc["t_star"].detach().max()) <= t_max + 1e-9
        assert float(res["t_star"].detach().max()) < t_max
        # Explicit rejection of the old [0, 1] bug
        assert float(bc["t_star"].detach().max()) < 0.5
        assert float(res["t_star"].detach().max()) < 0.5

    def test_nominal_shapes(self, cfg_full: DataConfig):
        ic = sample_ic(cfg_full)
        bc = sample_bc(cfg_full)
        res = sample_residual(cfg_full)
        assert ic["x_star"].shape == (2000, 1)
        assert bc["x_star"].shape == (2000, 1)
        assert res["x_star"].shape == (20000, 1)
