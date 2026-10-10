"""
==============================================================================
tests/test_inference.py — Rechargement d'un PINN depuis un checkpoint
==============================================================================

Un checkpoint qu'on ne sait pas relire est un checkpoint perdu : la
validation (Étape 4) et le démonstrateur (Étape 5) en dépendent.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from config import DataConfig, LossConfig, ModelConfig, TrainConfig
from src.inference import (
    data_config_from_dict,
    load_trained_model,
    predict_grid,
    predict_over_times,
)
from src.models import PINN
from src.sampling import sample_all
from src.trainer import PINNTrainer
from src.utils import save_checkpoint


@pytest.fixture
def cfg() -> DataConfig:
    return DataConfig(
        N_ic=64, N_bc=64, N_res=128, seed=3, device="cpu",
        obj_shape="square", obj_cx=0.4, obj_cy=0.6, obj_radius=0.1,
    )


def _train_tiny(cfg: DataConfig, tmp_path, activation: str = "tanh") -> PINN:
    model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=3, n_neurons=10, activation=activation))
    batches = sample_all(cfg)
    trainer = PINNTrainer(
        model, batches["ic"], batches["bc"], batches["res"], LossConfig(),
        TrainConfig(adam_epochs=3, lbfgs_iterations=1, early_stopping=False, scheduler="none"),
        device=torch.device("cpu"), checkpoint_dir=tmp_path, use_tqdm=False, verbose=False,
        metadata={"data_cfg": asdict(cfg)},
    )
    trainer.fit()
    trainer.restore_best_weights()
    return model


def _probe(n: int = 7):
    nodes = np.linspace(0.0, 1.0, n)
    return nodes, nodes


class TestArchitectureRoundTrip:
    def test_architecture_dict_rebuilds_same_shape(self, cfg: DataConfig) -> None:
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=3, n_neurons=10, activation="sin"))
        clone = PINN.from_architecture(model.architecture())
        assert clone.n_parameters == model.n_parameters
        assert clone.activation_name == "sin"

    @pytest.mark.parametrize("activation", ["tanh", "sin", "gelu", "softplus"])
    def test_checkpoint_reload_gives_identical_predictions(
        self, cfg: DataConfig, tmp_path, activation: str
    ) -> None:
        model = _train_tiny(cfg, tmp_path, activation)
        loaded, _, _ = load_trained_model(tmp_path / "last_model.pt")
        x, y = _probe()
        assert np.allclose(predict_grid(model, x, y, 0.05), predict_grid(loaded, x, y, 0.05), atol=1e-6)
        assert loaded.activation_name == activation

    def test_data_config_is_recovered(self, cfg: DataConfig, tmp_path) -> None:
        _train_tiny(cfg, tmp_path)
        _, recovered, payload = load_trained_model(tmp_path / "best_model.pt")
        assert recovered is not None
        assert (recovered.obj_shape, recovered.obj_cx, recovered.obj_cy, recovered.obj_radius) == (
            "square", 0.4, 0.6, 0.1,
        )
        assert recovered.t_star_max == pytest.approx(cfg.t_star_max)
        assert "architecture" in payload["extra"]

    def test_loaded_model_is_in_eval_mode_and_differentiable(self, cfg: DataConfig, tmp_path) -> None:
        """Le résidu PDE doit rester calculable sur un modèle rechargé."""
        from src.physics import pde_residual

        _train_tiny(cfg, tmp_path)
        loaded, _, _ = load_trained_model(tmp_path / "best_model.pt")
        assert not loaded.training
        res = sample_all(cfg)["res"]
        assert torch.isfinite(pde_residual(loaded, res["x_star"], res["y_star"], res["t_star"])).all()


class TestLegacyCheckpoints:
    def test_architecture_inferred_from_state_dict(self, cfg: DataConfig, tmp_path) -> None:
        """Ancien format : aucun 'architecture' dans extra."""
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=3, n_neurons=10))
        save_checkpoint(tmp_path / "old.pt", model, extra={})
        loaded, recovered, _ = load_trained_model(tmp_path / "old.pt")
        assert recovered is None
        assert loaded.n_hidden_layers == 3 and loaded.n_neurons == 10
        x, y = _probe()
        assert np.allclose(predict_grid(model, x, y, 0.02), predict_grid(loaded, x, y, 0.02), atol=1e-6)

    def test_normalization_bounds_are_inferred(self, tmp_path) -> None:
        cfg = DataConfig(alpha=1e-5)  # t*_max = 0.05 ≠ défaut
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=6))
        save_checkpoint(tmp_path / "old.pt", model, extra={})
        loaded, _, _ = load_trained_model(tmp_path / "old.pt")
        assert loaded.architecture()["input_upper"][2] == pytest.approx(cfg.t_star_max)

    def test_unrecognised_state_dict_fails_loudly(self, tmp_path) -> None:
        torch.save({"model_state_dict": {"foo": torch.zeros(1)}, "extra": {}}, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            load_trained_model(tmp_path / "bad.pt")

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_trained_model(tmp_path / "nope.pt")


class TestDataConfigFromDict:
    def test_none_and_empty(self) -> None:
        assert data_config_from_dict(None) is None
        assert data_config_from_dict({}) is None

    def test_unknown_fields_are_ignored(self) -> None:
        recovered = data_config_from_dict({"alpha": 1e-5, "champ_futur": 123})
        assert recovered.alpha == 1e-5

    def test_tuples_restored(self) -> None:
        recovered = data_config_from_dict({"x_star_range": [0.0, 1.0]})
        assert recovered.x_star_range == (0.0, 1.0)


class TestPredictHelpers:
    def test_predict_grid_shape_and_convention(self, cfg: DataConfig) -> None:
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8))
        x, y = np.linspace(0, 1, 5), np.linspace(0, 1, 9)
        grid = predict_grid(model, x, y, 0.03)
        assert grid.shape == (5, 9)  # (len(x), len(y)) : indexing "ij"
        xs, ys = torch.tensor([[x[2]]], dtype=torch.float32), torch.tensor([[y[7]]], dtype=torch.float32)
        point = model.predict(xs, ys, torch.tensor([[0.03]]))
        assert grid[2, 7] == pytest.approx(float(point), abs=1e-6)

    def test_predict_over_times_stacks(self, cfg: DataConfig) -> None:
        model = PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8))
        stack = predict_over_times(model, *_probe(), np.array([0.0, 0.05, 0.1]))
        assert stack.shape == (3, 7, 7)
