"""
==============================================================================
tests/test_trainer.py — Tests unitaires de la boucle d'entraînement (Étape 3)
==============================================================================

POURQUOI ces tests existent
---------------------------
L'Étape 3 introduit la machinery d'optimisation : Adam + scheduler,
L-BFGS + closure, pondération dynamique, early stopping, checkpoints.
Les bugs y sont SILENCIEUX : une closure qui ne remet pas les gradients
à zéro, un historique qui retient des tenseurs, un early stopping qui
ne se déclenche jamais — rien ne plante, le run converge juste mal ou
fait exploser la mémoire à l'epoch 8 000.

Invariants verrouillés ici :

    - fit() exécute bien les deux phases et remplit l'historique ;
    - l'historique ne contient QUE des types Python natifs (anti-fuite) ;
    - best_model.pt correspond au minimum de la métrique surveillée ;
    - l'early stopping s'arrête avant la fin du planning ;
    - les schémas adaptatifs produisent des poids bornés, normalisés,
      et ne changent rien avant leur première échéance ;
    - le checkpoint se recharge dans un modèle neuf aux prédictions
      identiques ;
    - get_device / plot_loss_history remplissent leur contrat.

Budgets volontairement minuscules (réseau 2×8, N_res=128) : toute la
suite tient en quelques secondes sur CPU.

Lancer :
    pytest tests/test_trainer.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from config import DataConfig, LossConfig, ModelConfig, TrainConfig
from src.models import PINN
from src.sampling import sample_all
from src.trainer import HISTORY_KEYS, FitResult, PINNTrainer
from src.utils import (
    get_device,
    load_checkpoint,
    plot_loss_history,
    save_checkpoint,
)


# ==========================================================================
# Fixtures — budgets minuscules pour des tests rapides
# ==========================================================================

@pytest.fixture
def cfg() -> DataConfig:
    """Configuration physique légère (CPU, petits budgets)."""
    return DataConfig(N_ic=64, N_bc=64, N_res=128, seed=7, device="cpu")


@pytest.fixture
def model(cfg: DataConfig) -> PINN:
    """Réseau miniature (2 couches × 8 neurones) non entraîné."""
    return PINN.from_config(cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=7))


@pytest.fixture
def batches(cfg: DataConfig) -> Dict[str, Dict[str, torch.Tensor]]:
    """Les trois familles de points de l'Étape 1."""
    return sample_all(cfg)


def _tiny_train_cfg(**overrides) -> TrainConfig:
    """
    TrainConfig minimal : 3 epochs Adam + 2 itérations L-BFGS, sans
    early stopping (les tests qui le ciblent le réactivent explicitement).
    """
    defaults = dict(
        adam_epochs=3,
        lbfgs_iterations=2,
        early_stopping=False,
        scheduler="none",
        log_every=1,
        lbfgs_log_every=1,
        checkpoint_every=2,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_trainer(
    model: PINN,
    batches: Dict[str, Dict[str, torch.Tensor]],
    tmp_path: Path,
    train_cfg: TrainConfig,
    loss_cfg: LossConfig | None = None,
) -> PINNTrainer:
    """Assemble un trainer silencieux écrivant dans tmp_path."""
    return PINNTrainer(
        model,
        batches["ic"],
        batches["bc"],
        batches["res"],
        loss_cfg or LossConfig(),
        train_cfg,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        use_tqdm=False,
        verbose=False,
    )


# ==========================================================================
# fit() — phases, historique, artefacts
# ==========================================================================

class TestFit:
    """Le contrat général de PINNTrainer.fit()."""

    def test_runs_both_phases_and_fills_history(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """3 epochs Adam puis 2 itérations L-BFGS → 5 lignes d'historique."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        result = trainer.fit()

        assert result.adam_epochs_run == 3
        assert result.lbfgs_iterations_run == 2
        assert result.stop_reason == "completed"
        assert result.early_stopped is False

        n_records = len(trainer.history["iteration"])
        assert n_records == 5
        # Les phases sont ordonnées et nommées : adam d'abord, lbfgs ensuite.
        assert trainer.history["phase"] == ["adam"] * 3 + ["lbfgs"] * 2
        # Les itérations globales sont consécutives (x-axis des figures).
        assert trainer.history["iteration"] == list(range(5))
        # Toutes les clés du contrat sont présentes et alignées.
        for key in HISTORY_KEYS:
            assert key in trainer.history
            assert len(trainer.history[key]) == n_records

    def test_losses_decrease_or_stay_finite(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """Chaque valeur enregistrée est finie et positive (MSE)."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        trainer.fit()

        for key in ("loss_total", "loss_ic", "loss_bc", "loss_res"):
            for value in trainer.history[key]:
                assert math.isfinite(value)
                assert value >= 0.0

    def test_history_stores_only_native_python_types(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """
        Anti-fuite mémoire : aucun torch.Tensor dans l'historique.

        Stocker des Tenseurs (même détachés) retiendrait des références
        aux graphes autograd et ferait croître la RAM à chaque epoch.
        C'est LE bug mémoire classique d'une boucle PINN.
        """
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        trainer.fit()

        for key, entries in trainer.history.items():
            for entry in entries:
                assert not isinstance(entry, torch.Tensor), (
                    f"{key} contient un torch.Tensor → fuite mémoire potentielle"
                )
                assert isinstance(entry, (str, int, float)), (
                    f"{key} contient un type non sérialisable : {type(entry)}"
                )

    def test_fit_twice_is_refused(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """Un trainer est mono-usage : le second fit() doit échouer bruyamment."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        trainer.fit()
        with pytest.raises(RuntimeError, match="déjà été exécuté"):
            trainer.fit()

    def test_artifacts_are_written(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """best_model.pt, last_model.pt et history.json existent et se lisent."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        result = trainer.fit()

        assert result.best_checkpoint.is_file()
        assert result.last_checkpoint.is_file()
        assert trainer.history_json_path.is_file()

        with open(trainer.history_json_path, encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        assert len(payload["history"]["iteration"]) == 5
        assert payload["weighting"] == "fixed"


# ==========================================================================
# Record & checkpoints
# ==========================================================================

class TestBestCheckpoint:
    """best_model.pt doit refléter le minimum de la métrique surveillée."""

    def test_best_loss_is_the_minimum_of_history(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        result = trainer.fit()

        assert result.best_loss == pytest.approx(min(trainer.history["loss_total"]))
        assert result.best_epoch == trainer.history["iteration"][
            trainer.history["loss_total"].index(min(trainer.history["loss_total"]))
        ]

    def test_best_checkpoint_restores_weights(
        self, model: PINN, batches, cfg: DataConfig, tmp_path: Path
    ) -> None:
        """Le best se recharge dans un modèle neuf → mêmes prédictions."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        result = trainer.fit()

        fresh_model = PINN.from_config(
            cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=1234)
        )
        payload = load_checkpoint(result.best_checkpoint, model=fresh_model)

        x_probe = torch.rand(32, 1)
        y_probe = torch.rand(32, 1)
        t_probe = torch.rand(32, 1) * cfg.t_star_max
        with torch.no_grad():
            restored = fresh_model.predict(x_probe, y_probe, t_probe)
        # Le trainer a continué après le record → on ne peut pas comparer
        # au modèle courant ; on vérifie la cohérence du payload.
        assert payload["best_loss"] == pytest.approx(result.best_loss)
        assert payload["model_class"] == "PINN"
        assert set(payload["model_state_dict"]) == set(model.state_dict())
        assert restored.shape == (32, 1)
        assert torch.isfinite(restored).all()

    def test_restore_best_weights_returns_false_before_fit(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """Sans record en mémoire, restore_best_weights() décline proprement."""
        trainer = _make_trainer(model, batches, tmp_path, _tiny_train_cfg())
        assert trainer.restore_best_weights() is False


# ==========================================================================
# Early stopping
# ==========================================================================

class TestEarlyStopping:
    """Filet de sécurité budgétaire : arrêt avant la fin du planning."""

    def test_stops_when_no_significant_improvement(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """
        min_delta énorme → aucune amélioration ne compte jamais : le run
        doit s'arrêter dès que la patience (2) est épuisée, c'est-à-dire
        après l'epoch 0 (record initial) + 2 epochs sans record.
        """
        train_cfg = _tiny_train_cfg(
            adam_epochs=20,
            lbfgs_iterations=0,
            early_stopping=True,
            es_patience=2,
            es_min_delta=1e9,
        )
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        result = trainer.fit()

        assert result.early_stopped is True
        assert result.stop_reason == "early_stopping"
        assert result.adam_epochs_run == 3  # epoch 0 (record) + 2 sans record
        assert len(trainer.history["iteration"]) == 3

    def test_does_not_trigger_when_disabled(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """Early stopping désactivé → le planning complet s'exécute."""
        train_cfg = _tiny_train_cfg(
            adam_epochs=5,
            lbfgs_iterations=0,
            early_stopping=False,
            es_patience=1,
            es_min_delta=1e9,
        )
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        result = trainer.fit()

        assert result.early_stopped is False
        assert result.adam_epochs_run == 5

    def test_monitor_residual_uses_pde_residual(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """es_monitor='residual' : le record suit L_res, pas L_total."""
        train_cfg = _tiny_train_cfg(adam_epochs=4, lbfgs_iterations=0)
        train_cfg.es_monitor = "residual"
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        result = trainer.fit()

        assert result.best_loss == pytest.approx(min(trainer.history["loss_res"]))


# ==========================================================================
# Pondération dynamique
# ==========================================================================

class TestDynamicWeighting:
    """Schémas grad_norm / lr_annealing : bornés, normalisés, programmés."""

    @pytest.mark.parametrize("scheme", ["grad_norm", "lr_annealing"])
    def test_factors_are_bounded_and_normalized(
        self, scheme: str, model: PINN, batches, tmp_path: Path
    ) -> None:
        """
        Quel que soit le schéma : poids strictement positifs, dans (0, 1],
        et max(w_ic, w_bc, w_res) = 1 à chaque mise à jour (normalisation
        qui garantit que la loss totale ne change pas d'échelle).
        """
        train_cfg = _tiny_train_cfg(
            adam_epochs=3,
            lbfgs_iterations=0,
            weighting=scheme,
            weighting_update_every=1,
        )
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        trainer.fit()

        weights = list(
            zip(trainer.history["w_ic"], trainer.history["w_bc"], trainer.history["w_res"])
        )
        # epoch 0 : poids de configuration intacts (aucune mise à jour)…
        assert weights[0] == (1.0, 1.0, 1.0)
        # …puis facteurs adaptatifs normalisés.
        for triplet in weights[1:]:
            assert all(0.0 < weight <= 1.0 + 1e-9 for weight in triplet)
            assert max(triplet) == pytest.approx(1.0, rel=1e-9)

    def test_weights_wait_for_their_schedule(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """update_every=5 avec 3 epochs : aucun facteur ne bouge."""
        train_cfg = _tiny_train_cfg(
            adam_epochs=3,
            lbfgs_iterations=0,
            weighting="grad_norm",
            weighting_update_every=5,
        )
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        trainer.fit()

        for triplet in zip(
            trainer.history["w_ic"], trainer.history["w_bc"], trainer.history["w_res"]
        ):
            assert triplet == (1.0, 1.0, 1.0)

    def test_config_weights_multiply_adaptive_factors(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """w_eff = w_config × w_adaptatif : ici w_ic=10 doit se lire."""
        train_cfg = _tiny_train_cfg(
            adam_epochs=2,
            lbfgs_iterations=0,
            weighting="grad_norm",
            weighting_update_every=1,
        )
        trainer = _make_trainer(
            model, batches, tmp_path, train_cfg, loss_cfg=LossConfig(w_ic=10.0)
        )
        trainer.fit()

        # La dernière ligne porte les poids effectifs après mise à jour.
        assert trainer.history["w_ic"][-1] > trainer.history["w_bc"][-1]
        assert trainer.history["w_ic"][-1] <= 10.0 + 1e-9


# ==========================================================================
# Scheduler Adam
# ==========================================================================

class TestScheduler:
    """Le lr suit bien le scheduler configuré."""

    def test_cosine_anneals_the_lr(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        """CosineAnnealingLR : le lr final doit être inférieur au lr initial."""
        train_cfg = _tiny_train_cfg(
            adam_epochs=4,
            lbfgs_iterations=0,
            scheduler="cosine",
            scheduler_min_lr=1e-6,
        )
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        trainer.fit()

        learning_rates = trainer.history["lr"]
        assert learning_rates[-1] < learning_rates[0]

    def test_none_keeps_lr_constant(
        self, model: PINN, batches, tmp_path: Path
    ) -> None:
        train_cfg = _tiny_train_cfg(adam_epochs=3, lbfgs_iterations=0)
        trainer = _make_trainer(model, batches, tmp_path, train_cfg)
        trainer.fit()

        assert len(set(trainer.history["lr"])) == 1


# ==========================================================================
# utils Étape 3 — device, checkpoints, figure
# ==========================================================================

class TestUtilsStep3:
    """Contrats de get_device / save+load_checkpoint / plot_loss_history."""

    def test_get_device_auto_returns_a_valid_device(self) -> None:
        device = get_device("auto")
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")

    def test_get_device_cpu_is_forced(self) -> None:
        assert get_device("cpu") == torch.device("cpu")

    def test_get_device_rejects_unknown_request(self) -> None:
        with pytest.raises(ValueError, match="Device inconnu"):
            get_device("tpu")

    def test_get_device_falls_back_when_unavailable(self) -> None:
        """'cuda' indisponible → fallback gracieux, pas d'exception."""
        device = get_device("cuda" if not torch.cuda.is_available() else "cpu")
        assert isinstance(device, torch.device)

    def test_save_and_load_checkpoint_roundtrip(
        self, model: PINN, cfg: DataConfig, tmp_path: Path
    ) -> None:
        """Un checkpoint sauvegardé puis rechargé redonne les mêmes poids."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Un VRAI pas d'optimisation (backward + step) pour peupler les
        # momenta d'Adam : sans gradient, Adam saute les paramètres et
        # son état reste vide — le roundtrip n'aurait rien à sauver.
        loss_probe = model(
            torch.rand(8, 1, requires_grad=True),
            torch.rand(8, 1, requires_grad=True),
            torch.rand(8, 1, requires_grad=True),
        ).sum()
        loss_probe.backward()
        optimizer.step()

        path = save_checkpoint(
            tmp_path / "roundtrip.pt",
            model,
            epoch=17,
            phase="adam",
            best_loss=0.5,
            optimizers={"adam": optimizer},
            history={"iteration": [0, 1], "loss_total": [1.0, 0.5]},
            extra={"seed": 42},
        )
        assert path.is_file()

        fresh_model = PINN.from_config(
            cfg, ModelConfig(n_hidden_layers=2, n_neurons=8, seed=99)
        )
        fresh_optimizer = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        payload = load_checkpoint(
            path, model=fresh_model, optimizers={"adam": fresh_optimizer}
        )

        assert payload["epoch"] == 17
        assert payload["best_loss"] == 0.5
        assert payload["extra"]["seed"] == 42

        x_probe = torch.rand(16, 1)
        y_probe = torch.rand(16, 1)
        t_probe = torch.rand(16, 1) * cfg.t_star_max
        with torch.no_grad():
            original = model.predict(x_probe, y_probe, t_probe)
            restored = fresh_model.predict(x_probe, y_probe, t_probe)
        assert torch.allclose(original, restored, atol=1e-6), (
            "Le checkpoint n'a pas restauré les poids à l'identique"
        )
        # L'état d'optimiseur a bien suivi (momenta présents).
        assert len(fresh_optimizer.state) > 0

    def test_load_missing_checkpoint_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "inexistant.pt")

    def test_plot_loss_history_writes_png(self, tmp_path: Path) -> None:
        """La figure s'écrit à partir d'un historique synthétique."""
        history: Dict[str, List] = {
            "phase": ["adam"] * 6 + ["lbfgs"] * 4,
            "iteration": list(range(10)),
            "loss_total": [10.0 * math.exp(-0.4 * i) for i in range(10)],
            "loss_ic": [1.0 * math.exp(-0.3 * i) for i in range(10)],
            "loss_bc": [0.5 * math.exp(-0.5 * i) for i in range(10)],
            "loss_res": [8.0 * math.exp(-0.2 * i) for i in range(10)],
            "w_ic": [1.0] * 10,
            "w_bc": [1.0] * 10,
            "w_res": [1.0] * 10,
            "lr": [1e-3 * (0.5 ** (i // 3)) for i in range(10)],
            "elapsed": [0.1 * (i + 1) for i in range(10)],
        }
        out_path = tmp_path / "courbes.png"
        resolved = plot_loss_history(
            history, out_path, best_epoch=7, early_stopped=False
        )
        assert resolved.is_file()
        assert resolved.stat().st_size > 0

    def test_plot_loss_history_rejects_empty_history(self, tmp_path: Path) -> None:
        history: Dict[str, List] = {key: [] for key in HISTORY_KEYS}
        with pytest.raises(ValueError, match="Historique vide"):
            plot_loss_history(history, tmp_path / "vide.png")
