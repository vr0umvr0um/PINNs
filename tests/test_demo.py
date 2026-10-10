"""
==============================================================================
tests/test_demo.py — Logique du démonstrateur interactif (Étape 5)
==============================================================================

On teste la logique sans navigateur : géométrie, conversions « gratuites »
(α, T_amb, T_obj), cache de la référence, rendu, avertissement quand
l'objet demandé n'est pas celui d'entraînement, ré-entraînement.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

import src.demo as demo
from config import DataConfig, ModelConfig
from src.demo import (
    compare_for_object,
    describe_object,
    format_duration,
    metrics_text,
    object_config,
    physical_time_horizon,
    reference_solution,
    render_snapshot,
    retrain_for_object,
    same_object,
    seconds_to_t_star,
    to_celsius,
)
from src.models import PINN


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig()


@pytest.fixture
def model(cfg: DataConfig) -> PINN:
    return PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8))


class TestObjectConfig:
    def test_valid_object(self, cfg: DataConfig) -> None:
        new = object_config(cfg, "square", 0.3, 0.7, 0.1)
        assert (new.obj_shape, new.obj_cx, new.obj_cy, new.obj_radius) == ("square", 0.3, 0.7, 0.1)
        assert cfg.obj_shape == "disk"  # la config de base n'est pas mutée

    @pytest.mark.parametrize(
        "shape, cx, cy, r",
        [
            ("disk", 0.05, 0.5, 0.15),   # sort de la pièce à gauche
            ("disk", 0.5, 0.95, 0.1),    # touche la paroi du haut
            ("square", 0.5, 0.5, 0.0),   # rayon nul
            ("triangle", 0.5, 0.5, 0.1), # forme inconnue
        ],
    )
    def test_invalid_objects_rejected(self, cfg: DataConfig, shape, cx, cy, r) -> None:
        with pytest.raises(ValueError):
            object_config(cfg, shape, cx, cy, r)

    def test_same_object(self, cfg: DataConfig) -> None:
        assert same_object(cfg, DataConfig())
        assert not same_object(cfg, object_config(cfg, "disk", 0.5, 0.5, 0.2))
        assert not same_object(cfg, object_config(cfg, "square", 0.5, 0.5, 0.15))

    def test_describe(self, cfg: DataConfig) -> None:
        assert "disque" in describe_object(cfg)
        assert "carré" in describe_object(object_config(cfg, "square", 0.5, 0.5, 0.1))


class TestFreePhysicalParameters:
    """α, T_amb, T_obj ne demandent aucun ré-entraînement."""

    def test_default_horizon_is_5000_seconds(self, cfg: DataConfig) -> None:
        assert physical_time_horizon(cfg, cfg.alpha) == pytest.approx(cfg.t_max)

    def test_slower_material_covers_longer_horizon(self, cfg: DataConfig) -> None:
        assert physical_time_horizon(cfg, cfg.alpha / 10) == pytest.approx(10 * cfg.t_max)

    def test_seconds_to_t_star_matches_config_conversion(self, cfg: DataConfig) -> None:
        assert seconds_to_t_star(cfg, cfg.alpha, 2500.0) == pytest.approx(cfg.to_t_star(2500.0))

    def test_t_star_is_clipped_to_training_domain(self, cfg: DataConfig) -> None:
        assert seconds_to_t_star(cfg, cfg.alpha, 1e9) == cfg.t_star_max
        assert seconds_to_t_star(cfg, cfg.alpha, -5.0) == 0.0

    def test_invalid_alpha(self, cfg: DataConfig) -> None:
        with pytest.raises(ValueError):
            physical_time_horizon(cfg, 0.0)

    def test_to_celsius(self) -> None:
        assert to_celsius(np.array([0.0, 1.0]), 20.0, 80.0).tolist() == [20.0, 80.0]

    def test_format_duration(self) -> None:
        assert format_duration(42) == "42 s"
        assert format_duration(725) == "12 min 05 s"
        assert format_duration(5000) == "1 h 23 min"


class TestReference:
    def test_reference_is_cached(self, cfg: DataConfig) -> None:
        first = reference_solution(cfg, 31, 6)
        assert reference_solution(cfg, 31, 6) is first

    def test_cache_distinguishes_objects(self, cfg: DataConfig) -> None:
        other = object_config(cfg, "disk", 0.4, 0.5, 0.15)
        assert reference_solution(cfg, 31, 6) is not reference_solution(other, 31, 6)

    def test_compare_for_object_shapes(self, cfg: DataConfig, model: PINN) -> None:
        report = compare_for_object(model, cfg, n_grid=21, n_times=5)
        assert report.prediction.shape == (5, 21, 21) == report.reference.shape


class TestRendering:
    def test_render_returns_figure_with_six_axes_incl_colorbars(self, cfg: DataConfig, model: PINN) -> None:
        report = compare_for_object(model, cfg, n_grid=21, n_times=5)
        figure = render_snapshot(report, cfg, 0.05, alpha=cfg.alpha, T_amb=20.0, T_obj=80.0)
        assert isinstance(figure, Figure)
        titles = [axis.get_title() for axis in figure.axes]
        assert "PINN" in titles
        assert any("Référence" in title for title in titles)
        assert any("erreur" in title.lower() for title in titles)
        # Le titre global affiche le temps PHYSIQUE.
        assert "t = " in figure._suptitle.get_text()

    def test_time_label_depends_on_alpha(self, cfg: DataConfig, model: PINN) -> None:
        report = compare_for_object(model, cfg, n_grid=21, n_times=5)
        slow = render_snapshot(report, cfg, 0.1, alpha=cfg.alpha / 10, T_amb=20, T_obj=80)
        assert "13 h 53 min" in slow._suptitle.get_text()  # 0.1·L²/(2e-6) = 50 000 s

    def test_metrics_warn_when_object_differs(self, cfg: DataConfig, model: PINN) -> None:
        other = object_config(cfg, "square", 0.3, 0.3, 0.1)
        report = compare_for_object(model, other, n_grid=21, n_times=5)
        same = metrics_text(report, 0.05, cfg, cfg, T_amb=20, T_obj=80)
        different = metrics_text(report, 0.05, other, cfg, T_amb=20, T_obj=80)
        assert "entraîné pour" not in same
        assert "entraîné pour" in different and "disque" in different


class TestRetrain:
    @pytest.fixture(autouse=True)
    def tiny_presets(self, monkeypatch) -> None:
        monkeypatch.setitem(demo.RETRAIN_PRESETS, "Mini", (4, 1, 32, 32, 64))

    def test_unknown_preset(self, cfg: DataConfig) -> None:
        with pytest.raises(ValueError):
            retrain_for_object(cfg, "Inexistant")

    def test_retrain_returns_usable_model_for_new_object(self, cfg: DataConfig, tmp_path) -> None:
        new_object = object_config(cfg, "square", 0.3, 0.6, 0.1)
        calls = []
        result = retrain_for_object(
            new_object, "Mini", progress=lambda f, m: calls.append(f), checkpoint_dir=str(tmp_path)
        )
        assert same_object(result.data_cfg, new_object)
        assert np.isfinite(result.final_losses["loss_total"])
        assert not result.model.training
        report = compare_for_object(result.model, new_object, n_grid=21, n_times=3)
        assert np.isfinite(report.l2_global)
        assert (tmp_path / "best_model.pt").is_file()

    def test_retrained_checkpoint_carries_its_object(self, cfg: DataConfig, tmp_path) -> None:
        from src.inference import load_trained_model

        new_object = object_config(cfg, "square", 0.3, 0.6, 0.1)
        retrain_for_object(new_object, "Mini", checkpoint_dir=str(tmp_path))
        _, recovered, _ = load_trained_model(tmp_path / "best_model.pt")
        assert same_object(recovered, new_object)


class TestGradioApp:
    def test_app_builds_and_refresh_works(self, cfg: DataConfig, model: PINN) -> None:
        pytest.importorskip("gradio")
        from src.gradio_app import build_app

        app = build_app(model, cfg, n_grid=21, n_times=5)
        assert app is not None
