"""
==============================================================================
tests/test_fd_solver.py — Solveur de référence aux différences finies (Étape 4)
==============================================================================

Un solveur de référence qui se trompe est pire que pas de solveur : on
« validerait » un PINN contre une fausse vérité. Ces tests le verrouillent
sur des propriétés mathématiques indépendantes du code :

    - mode propre exact sin(πx)sin(πy)e^{-2π²t} (erreur petite, ordre 2) ;
    - principe du maximum : 0 ≤ T* ≤ 1 pour une IC dans [0, 1] ;
    - Dirichlet : T* = 0 sur les 4 parois à tout instant ;
    - symétries d'un objet centré ; décroissance de l'énergie ;
    - instants demandés atteints exactement ;
    - comparaison à la série de Fourier du problème réel (disque).
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DataConfig
from src.fd_solver import STABILITY_RATIO, initial_field, solve_heat_fd


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig()


def _mode(n_grid: int) -> np.ndarray:
    nodes = np.linspace(0.0, 1.0, n_grid)
    grid_x, grid_y = np.meshgrid(nodes, nodes, indexing="ij")
    return np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)


def _eigenmode_error(cfg: DataConfig, n_grid: int) -> float:
    mode = _mode(n_grid)
    solution = solve_heat_fd(cfg, n_grid=n_grid, n_times=3, initial_condition=mode)
    exact = mode * np.exp(-2.0 * np.pi**2 * solution.t_star[-1])
    return float(np.linalg.norm(solution.T_star[-1] - exact) / np.linalg.norm(exact))


class TestAccuracy:
    def test_eigenmode_error_is_small(self, cfg: DataConfig) -> None:
        assert _eigenmode_error(cfg, 51) < 2e-3

    def test_second_order_convergence_in_space(self, cfg: DataConfig) -> None:
        """Diviser h par 2 doit diviser l'erreur par ~4 (ordre 2)."""
        coarse = _eigenmode_error(cfg, 26)
        fine = _eigenmode_error(cfg, 51)
        assert np.log2(coarse / fine) == pytest.approx(2.0, abs=0.3)

    def test_matches_fourier_series_for_the_disk(self, cfg: DataConfig) -> None:
        """Contre la solution en série de Fourier du VRAI problème (disque)."""
        fine_grid = 400
        centers = (np.arange(fine_grid) + 0.5) / fine_grid
        grid_x, grid_y = np.meshgrid(centers, centers, indexing="ij")
        indicator = ((grid_x - 0.5) ** 2 + (grid_y - 0.5) ** 2 <= 0.15**2).astype(float)
        modes = np.arange(1, 41)
        basis = np.sin(np.pi * np.outer(modes, centers))
        coefficients = 4.0 * (basis @ indicator @ basis.T) / fine_grid**2
        decay = np.pi**2 * (modes[:, None] ** 2 + modes[None, :] ** 2)

        solution = solve_heat_fd(cfg, n_grid=101, n_times=11)
        nodes = solution.x_star
        node_basis = np.sin(np.pi * np.outer(nodes, modes))
        k = 5  # t* = 0.05
        exact = node_basis @ (coefficients * np.exp(-decay * solution.t_star[k])) @ node_basis.T
        relative = np.linalg.norm(solution.T_star[k] - exact) / np.linalg.norm(exact)
        assert relative < 5e-3


class TestPhysics:
    def test_maximum_principle(self, cfg: DataConfig) -> None:
        solution = solve_heat_fd(cfg, n_grid=61, n_times=11)
        assert solution.T_star.min() >= -1e-12
        assert solution.T_star.max() <= 1.0 + 1e-12

    def test_dirichlet_walls_stay_zero(self, cfg: DataConfig) -> None:
        field = solve_heat_fd(cfg, n_grid=41, n_times=6).T_star
        for wall in (field[:, 0, :], field[:, -1, :], field[:, :, 0], field[:, :, -1]):
            assert np.all(wall == 0.0)

    def test_peak_temperature_decreases(self, cfg: DataConfig) -> None:
        field = solve_heat_fd(cfg, n_grid=61, n_times=11).T_star
        peaks = field.reshape(field.shape[0], -1).max(axis=1)
        assert np.all(np.diff(peaks) <= 1e-12)

    def test_total_heat_decreases(self, cfg: DataConfig) -> None:
        """Les parois à 0 évacuent la chaleur : ∫T* ne peut que décroître."""
        field = solve_heat_fd(cfg, n_grid=61, n_times=11).T_star
        heat = field.reshape(field.shape[0], -1).sum(axis=1)
        assert np.all(np.diff(heat) <= 1e-9)

    def test_centered_object_is_symmetric(self, cfg: DataConfig) -> None:
        last = solve_heat_fd(cfg, n_grid=61, n_times=5).T_star[-1]
        assert np.allclose(last, last[::-1, :], atol=1e-10)
        assert np.allclose(last, last[:, ::-1], atol=1e-10)
        assert np.allclose(last, last.T, atol=1e-10)

    def test_initial_snapshot_is_the_object_indicator(self, cfg: DataConfig) -> None:
        solution = solve_heat_fd(cfg, n_grid=41, n_times=3)
        assert np.array_equal(solution.T_star[0], initial_field(solution.x_star, solution.y_star, cfg))
        assert set(np.unique(solution.T_star[0])) == {0.0, 1.0}


class TestTimeGrid:
    def test_snapshots_hit_requested_times_exactly(self, cfg: DataConfig) -> None:
        wanted = [0.0, 0.013, 0.05, 0.1]
        solution = solve_heat_fd(cfg, n_grid=31, t_star_values=wanted)
        assert np.allclose(solution.t_star, wanted)
        assert solution.T_star.shape == (4, 31, 31)

    def test_default_grid_spans_zero_to_tmax(self, cfg: DataConfig) -> None:
        solution = solve_heat_fd(cfg, n_grid=21, n_times=7)
        assert solution.t_star[0] == 0.0
        assert solution.t_star[-1] == pytest.approx(cfg.t_star_max)

    def test_scheme_is_stable_at_the_cfl_limit(self, cfg: DataConfig) -> None:
        assert STABILITY_RATIO < 0.25  # limite théorique en 2D
        field = solve_heat_fd(cfg, n_grid=21, n_times=3).T_star
        assert np.isfinite(field).all()

    def test_at_returns_nearest_snapshot(self, cfg: DataConfig) -> None:
        solution = solve_heat_fd(cfg, n_grid=21, n_times=11)
        assert np.array_equal(solution.at(0.049), solution.T_star[5])


class TestValidation:
    def test_rejects_tiny_grid(self, cfg: DataConfig) -> None:
        with pytest.raises(ValueError):
            solve_heat_fd(cfg, n_grid=3)

    def test_rejects_unsorted_times(self, cfg: DataConfig) -> None:
        with pytest.raises(ValueError):
            solve_heat_fd(cfg, n_grid=11, t_star_values=[0.05, 0.01])

    def test_rejects_wrong_initial_condition_shape(self, cfg: DataConfig) -> None:
        with pytest.raises(ValueError):
            solve_heat_fd(cfg, n_grid=11, initial_condition=np.zeros((5, 5)))

    def test_square_object_supported(self) -> None:
        square = DataConfig(obj_shape="square", obj_cx=0.3, obj_cy=0.6, obj_radius=0.1)
        solution = solve_heat_fd(square, n_grid=41, n_times=3)
        hot = np.argwhere(solution.T_star[0] == 1.0)
        assert hot[:, 0].mean() * (1 / 40) == pytest.approx(0.3, abs=0.02)
        assert hot[:, 1].mean() * (1 / 40) == pytest.approx(0.6, abs=0.02)
