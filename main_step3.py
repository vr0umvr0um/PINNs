#!/usr/bin/env python3
"""
==============================================================================
main_step3.py — Étape 3 : Entraînement hybride Adam → L-BFGS
==============================================================================

Rôle de ce script dans le projet
--------------------------------
Point d'entrée de l'Étape 3. Il ASSEMBLE les brires validées des Étapes 1
et 2 (échantillonnage, réseau, autograd, loss) et les confie à
`PINNTrainer` (src/trainer.py), puis produit la figure
`step3_training_losses.png`.

    1. Construire les configurations (Data / Model / Loss / Train) depuis la CLI
    2. Sélectionner le device (cuda > mps > cpu) de façon transparente
    3. Générer les points de collocation            (Étape 1 — src.sampling)
    4. Construire le réseau T_θ                     (Étape 2 — src.models)
    5. Lancer l'entraînement hybride Adam → L-BFGS  (Étape 3 — src.trainer)
    6. Sauvegarder best_model.pt / last_model.pt / history.json
    7. Tracer les courbes de perte                   (src.utils.plot_loss_history)

LE MODE TEST (indispensable avant un run long)
----------------------------------------------
    python main_step3.py --test

réduit les budgets (500 epochs Adam + 50 itérations L-BFGS, échantillons
500/500/2000) et ajoute un BILAN MÉMOIRE : RSS (pic) avant/après le fit,
mémoire CUDA restituée après empty_cache, et vérification que
l'historique ne retient aucun tenseur PyTorch. Objectif : prouver
l'absence de bug ET de fuite mémoire AVANT d'engager un run de plusieurs
heures.

Usage
-----
    python main_step3.py                                   # run nominal
    python main_step3.py --test                            # run court de validation
    python main_step3.py --weighting grad_norm             # pondération dynamique
    python main_step3.py --weighting lr_annealing --scheduler cosine
    python main_step3.py --epochs-adam 10000 --iters-lbfgs 1000
    python main_step3.py --test --device cpu --no-tqdm
    python main_step3.py --resume checkpoints/last_model.pt  # repartir de poids sauvegardés
    python main_step3.py --tensorboard                     # logging TensorBoard (runs/)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from config import DataConfig, LossConfig, ModelConfig, TrainConfig
from src.models import PINN
from src.sampling import sample_all
from src.trainer import PINNTrainer
from src.utils import (
    format_seconds,
    get_device,
    load_checkpoint,
    plot_loss_history,
    set_seed,
)


# ==========================================================================
# CLI
# ==========================================================================

def parse_args() -> argparse.Namespace:
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(
        description=(
            "PINN Thermal 2D — Étape 3 : entraînement hybride Adam → L-BFGS "
            "avec suivi, checkpoints et early stopping."
        )
    )

    # --- Mode de run ------------------------------------------------------
    parser.add_argument(
        "--test", action="store_true",
        help=(
            "Run court de validation (500 epochs Adam + 50 itérations L-BFGS, "
            "budgets réduits 500/500/2000) avec bilan mémoire anti-fuite. "
            "À exécuter avant tout run long."
        ),
    )

    # --- Phase 1 : Adam ----------------------------------------------------
    parser.add_argument(
        "--epochs-adam", type=int, default=None,
        help="Nombre d'epochs Adam (défaut : TrainConfig=5000 ; --test : 500)",
    )
    parser.add_argument(
        "--adam-lr", type=float, default=None,
        help="Learning rate initial d'Adam (défaut : 1e-3)",
    )
    parser.add_argument(
        "--scheduler", choices=["plateau", "cosine", "none"], default=None,
        help="Scheduler de lr pour Adam (défaut : plateau)",
    )
    parser.add_argument(
        "--clip-grad", type=float, default=None,
        help="Clipping optionnel de la norme du gradient (défaut : désactivé)",
    )

    # --- Phase 2 : L-BFGS ---------------------------------------------------
    parser.add_argument(
        "--iters-lbfgs", type=int, default=None,
        help="Itérations L-BFGS, i.e. appels à step(closure) (défaut : 500 ; --test : 50)",
    )

    # --- Pondération de la loss ---------------------------------------------
    parser.add_argument(
        "--weighting", choices=["fixed", "grad_norm", "lr_annealing"], default="fixed",
        help=(
            "Stratégie de pondération (défaut : fixed). "
            "grad_norm : w_i ∝ 1/‖∇L_i‖ ; lr_annealing : Wang et al. 2021. "
            "Les schémas adaptatifs ne s'appliquent qu'en phase Adam."
        ),
    )
    parser.add_argument(
        "--weighting-every", type=int, default=None,
        help="Mise à jour des poids adaptatifs toutes les N epochs Adam (défaut : 100)",
    )
    parser.add_argument(
        "--weighting-warmup", type=int, default=None,
        help="Epochs Adam avant la première mise à jour des poids (défaut : 0)",
    )
    parser.add_argument(
        "--w-ic", type=float, default=1.0, help="Poids statique de L_ic (défaut : 1.0)",
    )
    parser.add_argument(
        "--w-bc", type=float, default=1.0, help="Poids statique de L_bc (défaut : 1.0)",
    )
    parser.add_argument(
        "--w-res", type=float, default=1.0, help="Poids statique de L_res (défaut : 1.0)",
    )

    # --- Early stopping -------------------------------------------------------
    parser.add_argument(
        "--no-early-stopping", action="store_true",
        help="Désactive l'early stopping (run jusqu'au bout du planning)",
    )
    parser.add_argument(
        "--monitor", choices=["total", "residual"], default=None,
        help="Métrique surveillée par l'early stopping / le best (défaut : total)",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Patience de l'early stopping en itérations (défaut : 1500)",
    )
    parser.add_argument(
        "--min-delta", type=float, default=None,
        help="Amélioration minimale absolue pour compter (défaut : 0)",
    )

    # --- Architecture (repris de l'Étape 2) ------------------------------------
    parser.add_argument(
        "--layers", type=int, default=None,
        help="Nombre de couches cachées (défaut : ModelConfig=4)",
    )
    parser.add_argument(
        "--neurons", type=int, default=None,
        help="Neurones par couche cachée (défaut : ModelConfig=64)",
    )
    parser.add_argument(
        "--activation", choices=["tanh", "sin", "gelu", "softplus"], default=None,
        help="Activation C² du réseau (défaut : tanh)",
    )

    # --- Échantillonnage (repris de l'Étape 1) ----------------------------------
    parser.add_argument(
        "--method", choices=["sobol", "lhs", "uniform"], default="sobol",
        help="Stratégie d'échantillonnage (défaut : sobol)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Graine de reproductibilité",
    )
    parser.add_argument(
        "--n-ic", type=int, default=None,
        help="Surcharge de N_ic (défaut : DataConfig=2000 ; --test : 500)",
    )
    parser.add_argument(
        "--n-bc", type=int, default=None,
        help="Surcharge de N_bc (défaut : DataConfig=2000 ; --test : 500)",
    )
    parser.add_argument(
        "--n-res", type=int, default=None,
        help="Surcharge de N_res (défaut : DataConfig=20000 ; --test : 2000)",
    )

    # --- Exécution ----------------------------------------------------------------
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto",
        help="Device PyTorch (défaut : auto = cuda > mps > cpu)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Dossier des artefacts best_model.pt / last_model.pt / history.json",
    )
    parser.add_argument(
        "--out", type=str, default="step3_training_losses.png",
        help="Chemin de la figure PNG de sortie",
    )
    parser.add_argument(
        "--tensorboard", action="store_true",
        help="Active le logging TensorBoard dans runs/ (paquet optionnel)",
    )
    parser.add_argument(
        "--no-tqdm", action="store_true",
        help="Désactive la barre de progression tqdm",
    )
    parser.add_argument(
        "--log-every", type=int, default=None,
        help="Fréquence des logs texte Adam (défaut : 100 epochs)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint à charger pour initialiser les poids (ex. last_model.pt)",
    )
    return parser.parse_args()


def build_configs_from_args(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[DataConfig, ModelConfig, LossConfig, TrainConfig]:
    """
    Construit les quatre dataclasses de configuration depuis la CLI.

    MODE TEST
    ---------
    `--test` réduit les DÉFAUTS (jamais les valeurs explicitement passées) :
    500 epochs Adam, 50 itérations L-BFGS, budgets 500/500/2000. C'est le
    run court recommandé pour valider l'absence de bugs et de fuites
    mémoire avant d'engager un vrai run long.

    Seuls les champs explicitement fournis (non-None) surchargent les
    valeurs par défaut : `python main_step3.py` sans aucun flag produit
    la configuration nominale du projet.
    """
    if args.test:
        if args.epochs_adam is None:
            args.epochs_adam = 500
        if args.iters_lbfgs is None:
            args.iters_lbfgs = 50
        if args.n_ic is None:
            args.n_ic = 500
        if args.n_bc is None:
            args.n_bc = 500
        if args.n_res is None:
            args.n_res = 2000

    data_kwargs: Dict[str, Any] = {"seed": args.seed, "device": str(device)}
    if args.n_ic is not None:
        data_kwargs["N_ic"] = args.n_ic
    if args.n_bc is not None:
        data_kwargs["N_bc"] = args.n_bc
    if args.n_res is not None:
        data_kwargs["N_res"] = args.n_res
    cfg = DataConfig(**data_kwargs)

    model_kwargs: Dict[str, Any] = {"seed": args.seed}
    if args.layers is not None:
        model_kwargs["n_hidden_layers"] = args.layers
    if args.neurons is not None:
        model_kwargs["n_neurons"] = args.neurons
    if args.activation is not None:
        model_kwargs["activation"] = args.activation
    model_cfg = ModelConfig(**model_kwargs)

    loss_cfg = LossConfig(w_ic=args.w_ic, w_bc=args.w_bc, w_res=args.w_res)

    train_kwargs: Dict[str, Any] = {
        "weighting": args.weighting,
        "early_stopping": not args.no_early_stopping,
    }
    if args.weighting_every is not None:
        train_kwargs["weighting_update_every"] = args.weighting_every
    if args.weighting_warmup is not None:
        train_kwargs["weighting_warmup"] = args.weighting_warmup
    if args.epochs_adam is not None:
        train_kwargs["adam_epochs"] = args.epochs_adam
    if args.iters_lbfgs is not None:
        train_kwargs["lbfgs_iterations"] = args.iters_lbfgs
    if args.adam_lr is not None:
        train_kwargs["adam_lr"] = args.adam_lr
    if args.scheduler is not None:
        train_kwargs["scheduler"] = args.scheduler
    if args.clip_grad is not None:
        train_kwargs["adam_clip_grad_norm"] = args.clip_grad
    if args.monitor is not None:
        train_kwargs["es_monitor"] = args.monitor
    if args.patience is not None:
        train_kwargs["es_patience"] = args.patience
    if args.min_delta is not None:
        train_kwargs["es_min_delta"] = args.min_delta
    if args.log_every is not None:
        train_kwargs["log_every"] = args.log_every
    train_cfg = TrainConfig(**train_kwargs)

    return cfg, model_cfg, loss_cfg, train_cfg


# ==========================================================================
# Bilan mémoire (mode --test)
# ==========================================================================

def _memory_snapshot() -> Dict[str, float]:
    """
    Photographie la mémoire du process.

    DEUX SONDES COMPLÉMENTAIRES
    ---------------------------
    - ru_maxrss (resource, Unix) : pic de mémoire RÉSIDENTE du process,
      en Ko sous Linux. C'est une marque haute MONOTONE : seule sa
      croissance entre deux instants est informative.
    - torch.cuda.memory_allocated : mémoire CUDA effectivement allouée
      par PyTorch (le tenseurs actifs, pas le cache réservé).

    POURQUOI CES SONDES ET PAS psutil ?
        resource est dans la stdlib — le bilan fonctionne partout sans
        dépendance supplémentaire.
    """
    snapshot: Dict[str, float] = {}
    try:
        import resource  # stdlib, Unix

        snapshot["rss_peak_mb"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss / 1024.0
    except (ImportError, AttributeError):  # pragma: no cover - Windows
        pass
    if torch.cuda.is_available():
        snapshot["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0**2)
        snapshot["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0**2)
    return snapshot


def _report_memory_test(
    trainer: PINNTrainer,
    before: Dict[str, float],
    device: torch.device,
) -> None:
    """
    Bilan anti-fuite mémoire du mode --test (à lire après fit()).

    Trois contrôles :
      1. l'historique ne retient AUCUN tenseur PyTorch (la fuite classique
         d'un PINN : stocker des Tensors détache leur graphe en RAM) ;
      2. la croissance du pic RSS reste modérée ;
      3. sur CUDA, la mémoire allouée revient proche du niveau d'avant le
         fit après empty_cache (les graphes libérés rendent la VRAM).
    """
    print("\n[test] Bilan mémoire — détection de fuite :")

    # 1. Types de l'historique
    leaky_keys = [
        key
        for key, entries in trainer.history.items()
        if any(isinstance(entry, torch.Tensor) for entry in entries)
    ]
    status = "OK" if not leaky_keys else "WARN"
    detail = "uniquement des types Python natifs" if not leaky_keys else f"tenseurs dans {leaky_keys}"
    print(f"  [{status}] historique : {detail}")

    # 2. Pic RSS (monotone : seul le Δ compte)
    after = _memory_snapshot()
    if "rss_peak_mb" in before and "rss_peak_mb" in after:
        growth = after["rss_peak_mb"] - before["rss_peak_mb"]
        status = "OK" if growth < 200.0 else "WARN"
        print(
            f"  [{status}] RSS (pic) : {before['rss_peak_mb']:.1f} Mo → "
            f"{after['rss_peak_mb']:.1f} Mo (Δ = {growth:+.1f} Mo)"
        )

    # 3. VRAM CUDA
    if device.type == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024.0**2)
        delta = allocated - before.get("cuda_allocated_mb", allocated)
        status = "OK" if abs(delta) < 50.0 else "WARN"
        print(
            f"  [{status}] CUDA alloué après empty_cache : {allocated:.1f} Mo "
            f"(Δ = {delta:+.1f} Mo vs avant fit)"
        )

    print("  (rappel : ru_maxrss est monotone — seul un Δ important signale une fuite)")


# ==========================================================================
# Orchestration
# ==========================================================================

def main() -> int:
    """
    Orchestration de l'Étape 3.

    Returns
    -------
    int
        Code retour shell (0 = succès).
    """
    args = parse_args()
    device = get_device(args.device)
    cfg, model_cfg, loss_cfg, train_cfg = build_configs_from_args(args, device)
    set_seed(cfg.seed)

    # ----- Bannière configuration ----------------------------------------
    if args.test:
        print("=" * 60)
        print("  MODE TEST — run court de validation (bugs + fuite mémoire)")
        print("=" * 60)
    print(cfg.summary())
    print(model_cfg.summary())
    print(f"  {loss_cfg.summary()}")
    print(train_cfg.summary())
    if device.type == "cuda":
        print(f"  [GPU] {torch.cuda.get_device_name(0)}")

    # ----- 1/4 Points de collocation (Étape 1) ---------------------------
    print("\n[1/4] Génération des points de collocation (Étape 1)…")
    batches = sample_all(cfg, method=args.method)
    print(
        f"  [OK] IC={cfg.N_ic}, BC={cfg.N_bc}, RES={cfg.N_res} "
        f"(méthode {args.method}, device={cfg.device})"
    )

    # ----- 2/4 Réseau (Étape 2) -------------------------------------------
    print("[2/4] Construction du réseau T_θ (Étape 2)…")
    model = PINN.from_config(cfg, model_cfg)
    print(model.summary())

    if args.resume is not None:
        payload = load_checkpoint(args.resume, model=model)
        print(
            f"  [OK] Poids restaurés depuis {args.resume} "
            f"(epoch={payload.get('epoch')}, best_loss={payload.get('best_loss')})"
        )

    # ----- 3/4 Entraînement hybride (Étape 3) ------------------------------
    print("[3/4] Entraînement hybride Adam → L-BFGS…")
    tensorboard_dir: Optional[Path] = None
    if args.tensorboard:
        from datetime import datetime

        tensorboard_dir = Path("runs") / f"step3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"  [info] Logging TensorBoard → {tensorboard_dir} (tensorboard --logdir runs/)")

    trainer = PINNTrainer(
        model,
        batches["ic"],
        batches["bc"],
        batches["res"],
        loss_cfg,
        train_cfg,
        device=device,
        checkpoint_dir=Path(args.checkpoint_dir),
        use_tqdm=not args.no_tqdm,
        tensorboard_dir=tensorboard_dir,
    )

    # Photographie mémoire AVANT le fit (après imports/matplotlib & données :
    # seuls les effets de l'entraînement sont mesurés).
    memory_before = _memory_snapshot()

    result = trainer.fit()

    # ----- 4/4 Bilan, mémoire (test), figure --------------------------------
    print("\n[4/4] Bilan & export de la figure…")
    print(result.summary())

    if args.test:
        _report_memory_test(trainer, memory_before, device)

    figure_path = plot_loss_history(
        trainer.history,
        Path(args.out),
        title=(
            "Étape 3 — Entraînement hybride Adam → L-BFGS\n"
            f"({model.n_parameters:,} paramètres, pondération : {train_cfg.weighting})"
        ),
        best_epoch=result.best_epoch,
        early_stopped=result.early_stopped,
    )
    print(f"  [OK] Figure sauvegardée → {figure_path}")

    # ----- Récapitulatif ------------------------------------------------------
    print("\n[confirm] Récapitulatif Étape 3 :")
    print(f"  Planning       : Adam {result.adam_epochs_run} epochs → "
          f"L-BFGS {result.lbfgs_iterations_run} itérations "
          f"({result.lbfgs_closure_evals} évaluations de closure)")
    print(f"  Durée          : {format_seconds(result.elapsed_seconds)}")
    print(f"  Record         : {result.best_loss:.6e} (itération {result.best_epoch})")
    print(f"  Best / Last    : {result.best_checkpoint.name} / {result.last_checkpoint.name} "
          f"dans {result.best_checkpoint.parent.resolve()}")
    print(f"  Historique     : {trainer.history_json_path.resolve()}")

    print("\n✓ Étape 3 terminée avec succès.")
    print("  → Prochaine étape : démonstrateur interactif Gradio (Étape 4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
