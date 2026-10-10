"""
==============================================================================
tests/test_validation.py — Métriques MSE / L2 relative (Étape 4)
==============================================================================
"""

from __future__ import annotations

import numpy as np
import pytest

from config import DataConfig, ModelConfig
from src.fd_solver import solve_heat_fd
from src.models import PINN
from src.validation import (
    compare_fields,
    metrics_at_times,
    mse,
    plot_validation,
    relative_l2_error,
    validate_pinn,
)


class TestScalarMetrics:
    def test_mse_known_value(self) -> None:
        assert mse(np.array([1.0, 3.0]), np.array([0.0, 1.0])) == pytest.approx((1 + 4) / 2)

    def test_relative_l2_known_value(self) -> None:
        # ‖(3,4)−(0,0)‖ / ‖(0,0)‖ n'a pas de sens ; on prend une réf non nulle.
        reference = np.array([3.0, 4.0])
        prediction = np.array([3.0, 4.0]) * 1.1
        assert relative_l2_error(prediction, reference) == pytest.approx(0.1)

    def test_zero_error_for_identical_fields(self) -> None:
        field = np.random.default_rng(0).random((5, 5))
        assert mse(field, field) == 0.0
        assert relative_l2_error(field, field) == 0.0

    def test_zero_reference_does_not_divide_by_zero(self) -> None:
        assert np.isfinite(relative_l2_error(np.ones((3, 3)), np.zeros((3, 3))))

    def test_relative_error_is_scale_invariant(self) -> None:
        rng = np.random.default_rng(1)
        reference, noise = rng.random((6, 6)) + 1.0, rng.random((6, 6)) * 0.1
        assert relative_l2_error(reference + noise, reference) == pytest.approx(
            relative_l2_error(10 * (reference + noise), 10 * reference)
        )


class TestCompareFields:
    def test_shapes_and_per_time_arrays(self) -> None:
        reference = np.random.default_rng(2).random((4, 6, 6))
        report = compare_fields(np.linspace(0, 0.1, 4), reference * 1.05, reference)
        assert report.mse_per_time.shape == (4,)
        assert report.l2_per_time == pytest.approx(np.full(4, 0.05))
        assert report.l2_global == pytest.approx(0.05)

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            compare_fields(np.zeros(2), np.zeros((2, 4, 4)), np.zeros((2, 5, 5)))

    def test_counts_negative_predictions(self) -> None:
        reference = np.ones((1, 2, 2))
        prediction = np.array([[[-0.1, 0.5], [0.5, 0.5]]])
        report = compare_fields(np.zeros(1), prediction, reference)
        assert report.negative_fraction == pytest.approx(0.25)

    def test_global_error_is_dominated_by_large_fields(self) -> None:
        """Norme espace-temps : un instant très froid pèse peu."""
        hot, cold = np.ones((4, 4)), np.full((4, 4), 1e-3)
        reference = np.stack([hot, cold])
        prediction = np.stack([hot, cold * 2])  # 100 % d'erreur à l'instant froid
        report = compare_fields(np.array([0.0, 0.1]), prediction, reference)
        assert report.l2_per_time[1] == pytest.approx(1.0)
        assert report.l2_global < 1e-3

    def test_summary_and_dict(self) -> None:
        reference = np.random.default_rng(3).random((2, 4, 4)) + 0.5
        report = compare_fields(np.array([0.0, 0.1]), reference * 0.9, reference)
        assert "L2" in report.summary(delta_T=60.0)
        assert set(report.as_dict()) >= {"mse_global", "l2_global", "l2_per_time"}

    def test_metrics_at_times_picks_nearest(self) -> None:
        reference = np.random.default_rng(4).random((3, 3, 3)) + 0.5
        report = compare_fields(np.array([0.0, 0.05, 0.1]), reference * 1.1, reference)
        rows = metrics_at_times(report, [0.049])
        assert rows[0]["t_star"] == pytest.approx(0.05)


class TestValidatePinn:
    def test_runs_on_untrained_network(self) -> None:
        cfg = DataConfig()
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8))
        reference = solve_heat_fd(cfg, n_grid=21, n_times=4)
        report = validate_pinn(model, reference)
        assert report.prediction.shape == reference.T_star.shape
        assert np.isfinite(report.l2_global)
        assert report.l2_global > 0.5  # un réseau aléatoire ne peut pas être bon

    def test_plot_creates_png(self, tmp_path) -> None:
        cfg = DataConfig()
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8))
        report = validate_pinn(model, solve_heat_fd(cfg, n_grid=21, n_times=11))
        path = plot_validation(report, tmp_path / "val.png")
        assert path.is_file() and path.stat().st_size > 5_000
