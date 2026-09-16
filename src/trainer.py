"""
==============================================================================
src/trainer.py — Boucle d'entraînement hybride Adam → L-BFGS (Étape 3)
==============================================================================

RÔLE DE CE MODULE
-----------------
L'Étape 1 a produit les points de collocation, l'Étape 2 le réseau T_θ,
le résidu autograd et la loss multi-objectif — le tout VALIDÉ. Ce module
met cette machinerie en mouvement : c'est ici que θ cesse d'être une
initialisation Xavier pour devenir une SOLUTION de l'équation de la
chaleur adimensionnée.

LA STRATÉGIE HYBRIDE, POURQUOI DEUX PHASES ?
--------------------------------------------
Phase 1 — Adam (exploration robuste)
    Adam normalise son pas par des moyennes mobiles des gradients : il
    tolère les gradients raides et bruités des DÉRIVÉES SECONDES du
    résidu, et s'échappe des mauvais bassins d'attraction. Mais son pas
    par-paramètre fait qu'il oscille indéfiniment AUTOUR du minimum sans
    jamais s'y installer (plafond typique ~1e-4 / 1e-5 sur la loss).

Phase 2 — L-BFGS (affinage quasi-Newton)
    L-BFGS mémorise les `history_size` dernières paires (pas, variation
    de gradient) pour reconstruire une approximation INVERSE de la
    Hessienne sans jamais la stocker explicitement. Son pas devient de
    plus en plus pertinent à mesure qu'on approche du minimum →
    convergence quasi exacte sur les résidus. En contrepartie, loin du
    minimum son modèle de courbure est trompeur et la line search peut
    échouer : d'où l'ordre Adam PUIS L-BFGS, jamais l'inverse.

LA CLOSURE L-BFGS (le piège n°1 de l'étape)
-------------------------------------------
`torch.optim.LBFGS.step()` ne consomme pas un gradient pré-calculé :
il exige une FONCTION `closure()` qui (1) remet les gradients à zéro,
(2) recalcule la loss depuis les poids COURANTS, (3) lance backward et
(4) RENVOIE la loss. L-BFGS l'appelle autant de fois que la line search
`strong_wolfe` évalue des points d'essai. La loss doit donc être
recalculée à l'intérieur de la closure — jamais mise en cache.

POURQUOI FULL-BATCH (aucun mini-batch)
---------------------------------------
L-BFGS mémorise une courbure ENTRE deux évaluations de la loss : si
l'objectif changeait entre deux appels de closure (mini-batchs
aléatoires), la mémoire de courbure deviendrait incohérente et la line
search échouerait. Tout le budget (N_ic, N_bc, N_res) est donc évalué à
chaque itération — c'est aussi ce qui rend le run parfaitement
déterministe et reproductible à seed fixée.

POURQUOI LES POIDS DYNAMIQUES SONT GELÉS EN PHASE L-BFGS
---------------------------------------------------------
Même raison : L-BFGS minimise un objectif FIXE. Changer w_ic / w_bc /
w_res en cours de phase déplacerait la cible sous les pieds de
l'optimiseur et invaliderait sa mémoire de courbure. Les schémas
auto-adaptatifs (grad_norm, lr_annealing) s'appliquent donc pendant la
phase Adam uniquement ; L-BFGS hérite des poids effectifs du dernier
pas Adam.

ANTI-FUITE MÉMOIRE (à savoir expliquer)
---------------------------------------
L'historique ne stocke QUE des float / str Python (via LossTerms.as_floats) :
stocker des Tenseurs y retiendrait le graphe autograd entier de chaque
epoch en mémoire → explosion de la RAM/VRAM après quelques milliers
d'itérations. Le meilleur état est copié sur CPU au moment du record ;
les graphes des itérations intermédiaires sont libérés par backward().
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from config import DEFAULT_LOSS_CONFIG, DEFAULT_TRAIN_CONFIG, LossConfig, TrainConfig
from src.losses import LossTerms, pinn_loss
from src.models import PINN
from src.utils import format_seconds, get_device, save_checkpoint

# Un batch de collocation = dict nom → tenseur (sortie de src.sampling).
CollocationBatch = Dict[str, torch.Tensor]

# Clés de l'historique d'entraînement (toutes des listes de floats, sauf
# "phase" qui est une liste de chaînes "adam" / "lbfgs").
HISTORY_KEYS: Tuple[str, ...] = (
    "phase",
    "iteration",
    "loss_total",
    "loss_ic",
    "loss_bc",
    "loss_res",
    "w_ic",
    "w_bc",
    "w_res",
    "lr",
    "elapsed",
)

# tqdm est optionnel : le trainer doit fonctionner sans (logging texte).
try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover - dépend de l'environnement
    _tqdm = None  # type: ignore[assignment]


# ==========================================================================
# Bilan d'entraînement
# ==========================================================================

@dataclass
class FitResult:
    """
    Résultat complet d'un appel à `PINNTrainer.fit()`.

    POURQUOI CE CONTENEUR plutôt que de renvoyer juste l'historique ?
        Un run d'entraînement se résume par des questions précises :
        « combien d'itérations ont réellement tourné ? », « le run
        s'est-il arrêté avant la fin ? pourquoi ? », « où est le
        meilleur checkpoint ? ». Regrouper ces réponses évite à
        l'appelant (main_step3, un notebook, un test) de les déduire
        de l'historique brin par brin.

    Attributs
    ---------
    history : dict[str, list]
        Historique complet (clés de `HISTORY_KEYS`), floats Python natifs.
    best_loss : float
        Meilleure valeur de la métrique surveillée (inf si 0 itération).
    best_epoch : int
        Itération (globale) du record (-1 si aucune).
    adam_epochs_run, lbfgs_iterations_run : int
        Itérations effectivement exécutées dans chaque phase (≤ planning).
    lbfgs_closure_evals : int
        Nombre total d'évaluations de closure (le vrai coût de L-BFGS :
        une line search peut évaluer plusieurs points d'essai par pas).
    final_losses : dict[str, float]
        Losses évaluées sur les poids EN FIN de run (après restauration
        éventuelle du best en cas de divergence).
    early_stopped : bool
        True si le run s'est interrompu avant la fin du planning
        (early stopping, divergence, erreur L-BFGS).
    stop_reason : str
        "completed" | "early_stopping" | "divergence_nan" | "lbfgs_error".
    elapsed_seconds : float
        Durée totale du fit.
    best_checkpoint, last_checkpoint : Path
        Chemins des deux checkpoints écrits.
    """

    history: Dict[str, List[Any]]
    best_loss: float
    best_epoch: int
    adam_epochs_run: int
    lbfgs_iterations_run: int
    lbfgs_closure_evals: int
    final_losses: Dict[str, float]
    early_stopped: bool
    stop_reason: str
    elapsed_seconds: float
    best_checkpoint: Path
    last_checkpoint: Path

    def summary(self) -> str:
        """Résumé textuel multi-lignes (pour le terminal en fin de run)."""
        separator = "=" * 60
        best_loss_txt = f"{self.best_loss:.6e}" if math.isfinite(self.best_loss) else "n/a"
        lines = [
            separator,
            "  PINNTrainer — bilan d'entraînement",
            separator,
            f"  Phases         : Adam {self.adam_epochs_run} epochs → "
            f"L-BFGS {self.lbfgs_iterations_run} itérations "
            f"({self.lbfgs_closure_evals} évaluations de closure)",
            f"  Durée totale   : {format_seconds(self.elapsed_seconds)}",
            f"  Record         : {best_loss_txt} à l'itération {self.best_epoch}",
            f"  Losses finales : L={self.final_losses['loss_total']:.3e} | "
            f"ic={self.final_losses['loss_ic']:.3e} | "
            f"bc={self.final_losses['loss_bc']:.3e} | "
            f"res={self.final_losses['loss_res']:.3e}",
            f"  Arrêt          : {self.stop_reason}"
            + ("  (avant la fin du planning)" if self.early_stopped else ""),
            f"  Checkpoints    : best → {self.best_checkpoint}",
            f"                   last → {self.last_checkpoint}",
            separator,
        ]
        return "\n".join(lines)


# ==========================================================================
# Le trainer
# ==========================================================================

class PINNTrainer:
    """
    Encapsule TOUT le cycle d'optimisation du PINN :

        1. Phase Adam   : scheduler de lr (plateau / cosine), pondération
                          dynamique optionnelle, early stopping ;
        2. Phase L-BFGS : closure strong_wolfe, objectif gelé ;
        3. Surveillance : historique complet, tqdm / TensorBoard ;
        4. Persistance  : best_model.pt (record), last_model.pt (reprise),
                          history.json (post-mortem).

    LE CONTRAT AVEC LES ÉTAPES 1-2
    -------------------------------
        model       : un PINN (src.models) déjà construit ;
        *_batch     : les dicts de tenseurs produits par src.sampling
                      (shapes (N, 1), requires_grad=True sur x*, y*, t*) ;
        loss_cfg    : pondérations STATIQUES de base (config) ;
        train_cfg   : planning hybride, early stopping, pondération
                      dynamique (config.TrainConfig).

    EXEMPLE
    --------
        >>> trainer = PINNTrainer(model, ic, bc, res, loss_cfg, train_cfg,
        ...                       device="auto", checkpoint_dir="checkpoints")
        >>> result = trainer.fit()
        >>> print(result.summary())
        >>> trainer.restore_best_weights()   # revenir au record

    Notes
    -----
    * `fit()` ne peut être appelé qu'une fois par instance ; pour
      poursuivre un run, reconstruire un trainer et restaurer les poids
      via `src.utils.load_checkpoint`.
    * `evaluate()` n'utilise pas `torch.no_grad()` : le résidu EXIGE le
      graphe autograd pour ses dérivées (un forward sous no_grad est
      un tenseur « mort », cf. docstring de PINN.predict). Seules les
      VALEURS sont extraites, les graphes sont libérés aussitôt.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: PINN,
        ic_batch: CollocationBatch,
        bc_batch: CollocationBatch,
        res_batch: CollocationBatch,
        loss_cfg: Optional[LossConfig] = None,
        train_cfg: Optional[TrainConfig] = None,
        *,
        device: Optional[Union[str, torch.device]] = None,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        use_tqdm: bool = True,
        tensorboard_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model : PINN
            Réseau T_θ (déjà instancié, ex. PINN.from_config).
        ic_batch, bc_batch, res_batch : dict[str, torch.Tensor]
            Sorties de sample_ic / sample_bc / sample_residual (Étape 1).
            Elles sont déplacées sur `device` si nécessaire.
        loss_cfg : LossConfig, optional
            Pondérations statiques de base. None → DEFAULT_LOSS_CONFIG.
        train_cfg : TrainConfig, optional
            Planning hybride + options. None → DEFAULT_TRAIN_CONFIG.
        device : str or torch.device, optional
            "auto" (défaut : cuda > mps > cpu), ou un device explicite.
        checkpoint_dir : str or Path
            Dossier des artefacts (best_model.pt, last_model.pt,
            history.json). Créé si absent.
        use_tqdm : bool
            Barre de progression tqdm (si installée).
        tensorboard_dir : str or Path, optional
            Si fourni, logging TensorBoard (nécessite le paquet
            `tensorboard` ; sinon dégradation gracieuse en message).
        verbose : bool
            Affichage des lignes de log pendant fit().
        """
        self.model = model
        self.loss_cfg = loss_cfg if loss_cfg is not None else DEFAULT_LOSS_CONFIG
        self.train_cfg = train_cfg if train_cfg is not None else DEFAULT_TRAIN_CONFIG
        self.verbose = verbose

        # --- Device -----------------------------------------------------
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = get_device(device if device is not None else "auto")
        self.model.to(self.device)
        # .to() sur un nn.Module déplace les paramètres EN PLACE : les
        # optimiseurs construits ci-dessous référencent les bons tenseurs.
        self.ic_batch = {key: value.to(self.device) for key, value in ic_batch.items()}
        self.bc_batch = {key: value.to(self.device) for key, value in bc_batch.items()}
        self.res_batch = {key: value.to(self.device) for key, value in res_batch.items()}

        # --- Optimiseurs (construits une fois, états dans les checkpoints) --
        self.adam_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_cfg.adam_lr
        )
        self.lbfgs_optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.train_cfg.lbfgs_lr,
            max_iter=self.train_cfg.lbfgs_max_iter,
            history_size=self.train_cfg.lbfgs_history_size,
            tolerance_grad=self.train_cfg.lbfgs_tolerance_grad,
            tolerance_change=self.train_cfg.lbfgs_tolerance_change,
            line_search_fn=self.train_cfg.lbfgs_line_search_fn,
        )
        self._build_scheduler()

        # --- Pondération effective ---------------------------------------
        # facteurs adaptatifs (1, 1, 1) tant qu'aucun schéma n'agit ;
        # poids effectifs = poids de config × facteurs adaptatifs.
        self._adaptive: Tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._effective_loss_cfg = copy.deepcopy(self.loss_cfg)

        # --- Historique & suivi ------------------------------------------
        self.history: Dict[str, List[Any]] = {key: [] for key in HISTORY_KEYS}
        self._global_iteration = 0
        self._current_phase = "adam"
        self._elapsed = 0.0
        self._start_time = 0.0

        # suivi du record (métrique surveillée, cf. TrainConfig.es_monitor)
        self._best_metric = math.inf
        self._best_epoch = -1
        self._epochs_without_improvement = 0
        self._best_state_cpu: Optional[Dict[str, torch.Tensor]] = None

        # compteurs de phases
        self._adam_epochs_run = 0
        self._lbfgs_iterations_run = 0
        self._closure_evals = 0

        # arrêt
        self._stop_reason = "completed"

        # --- Artefacts ----------------------------------------------------
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.last_checkpoint_path = self.checkpoint_dir / "last_model.pt"
        self.history_json_path = self.checkpoint_dir / "history.json"

        # --- Progression & TensorBoard ------------------------------------
        self.use_tqdm = use_tqdm and _tqdm is not None
        if use_tqdm and _tqdm is None and self.verbose:
            print("[info] tqdm n'est pas installé → logging texte uniquement (pip install tqdm).")
        self._active_bar: Optional[Any] = None

        self.writer: Optional[Any] = None
        if tensorboard_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            except ImportError:
                if self.verbose:
                    print(
                        "[info] tensorboard n'est pas installé → logging TensorBoard "
                        "désactivé (pip install tensorboard)."
                    )

        self._fitted = False

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def fit(self) -> FitResult:
        """
        Exécute le planning complet : phase Adam puis phase L-BFGS.

        Returns
        -------
        FitResult
            Bilan complet (historique, record, checkpoints, raison d'arrêt).

        Raises
        ------
        RuntimeError
            Si fit() a déjà été exécuté sur cette instance.
        """
        if self._fitted:
            raise RuntimeError(
                "PINNTrainer.fit() a déjà été exécuté sur cette instance. "
                "Créer un nouveau trainer — et éventuellement restaurer les "
                "poids via src.utils.load_checkpoint — pour poursuivre."
            )
        self._fitted = True
        self._start_time = time.perf_counter()
        self._elapsed = 0.0

        self._log(
            f"[fit] device={self.device.type} | Adam {self.train_cfg.adam_epochs} epochs "
            f"→ L-BFGS {self.train_cfg.lbfgs_iterations} itérations | "
            f"pondération={self.train_cfg.weighting} | "
            f"early stopping={'activé' if self.train_cfg.early_stopping else 'désactivé'}"
        )

        if self.train_cfg.adam_epochs > 0:
            self._train_adam()
        if self._stop_reason == "completed" and self.train_cfg.lbfgs_iterations > 0:
            self._train_lbfgs()

        # Évaluation finale sur les poids courants (après restauration du
        # best en cas de divergence : final == best dans ce cas).
        final_losses = self.evaluate()

        # Derniers artefacts
        self._save_checkpoint_file(self.last_checkpoint_path, include_history=True)
        self.save_history_json()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        return FitResult(
            history=self.history,
            best_loss=self._best_metric,
            best_epoch=self._best_epoch,
            adam_epochs_run=self._adam_epochs_run,
            lbfgs_iterations_run=self._lbfgs_iterations_run,
            lbfgs_closure_evals=self._closure_evals,
            final_losses=final_losses,
            early_stopped=self._stop_reason != "completed",
            stop_reason=self._stop_reason,
            elapsed_seconds=time.perf_counter() - self._start_time,
            best_checkpoint=self.best_checkpoint_path,
            last_checkpoint=self.last_checkpoint_path,
        )

    def evaluate(self) -> Dict[str, float]:
        """
        Évalue les trois termes de la loss sur les poids COURANTS.

        Différence avec l'entraînement : create_graph=False — le résidu
        n'a pas besoin d'être différentiable par rapport à θ, on ne
        veut que ses VALEURS (diagnostic, bilan de fin de run).

        Returns
        -------
        dict[str, float]
            {"loss_total", "loss_ic", "loss_bc", "loss_res"} en floats natifs.
        """
        terms = self._compute_terms(create_graph=False)
        return self._terms_to_values(terms)

    def restore_best_weights(self) -> bool:
        """
        Recharge dans le modèle les poids du meilleur checkpoint en mémoire.

        Returns
        -------
        bool
            True si une restauration a eu lieu, False s'il n'y a pas encore
            de record (fit() jamais lancé, ou 0 itération).
        """
        if self._best_state_cpu is None:
            return False
        self.model.load_state_dict(self._best_state_cpu)
        return True

    def save_history_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Exporte l'historique complet en JSON (post-mortem, rapports, plots).

        Parameters
        ----------
        path : str or Path, optional
            Chemin de sortie ; défaut `<checkpoint_dir>/history.json`.

        Returns
        -------
        Path
            Chemin résolu du fichier écrit.
        """
        target = Path(path) if path is not None else self.history_json_path
        payload = {
            "history": self.history,
            "best_loss": self._best_metric if math.isfinite(self._best_metric) else None,
            "best_epoch": self._best_epoch,
            "adam_epochs_run": self._adam_epochs_run,
            "lbfgs_iterations_run": self._lbfgs_iterations_run,
            "lbfgs_closure_evals": self._closure_evals,
            "stop_reason": self._stop_reason,
            "effective_weights": self._effective_loss_cfg.as_tuple(),
            "weighting": self.train_cfg.weighting,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        return target.resolve()

    # ------------------------------------------------------------------
    # Phase 1 — Adam
    # ------------------------------------------------------------------

    def _train_adam(self) -> None:
        """
        Boucle d'entraînement Adam : exploration robuste + scheduler de lr.

        Ordre des opérations PAR EPOCH (l'ordre est important) :
            1. zero_grad            — sinon les gradients s'ACCUMULENT
            2. forward + résidu     — un seul graphe autograd partagé
            3. pondération          — mise à jour éventuelle des poids
                                      adaptatifs (autograd.grad, graph intact)
            4. total.backward()     — libère le graphe
            5. clip optionnel + Adam.step()
            6. scheduler.step()     — plateau reçoit la loss courante
            7. record / best / early stopping / checkpoint
        """
        self._current_phase = "adam"
        n_epochs = self.train_cfg.adam_epochs
        progress = self._progress(range(n_epochs), total=n_epochs, desc="Adam", unit="epoch")

        try:
            for epoch in progress:
                iteration = self._global_iteration

                self.adam_optimizer.zero_grad(set_to_none=True)
                terms = self._compute_terms()

                if self._should_update_weights(epoch):
                    self._update_adaptive_weights(terms)

                # Total recalculé à la main : si les poids viennent de changer,
                # il faut pondérer le MÊME forward (terms.total est périmé).
                total = self._weighted_total(terms)
                total.backward()

                if self.train_cfg.adam_clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.train_cfg.adam_clip_grad_norm
                    )
                self.adam_optimizer.step()

                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # Le plateau s'adapte à la VALEUR de la loss...
                        self.scheduler.step(float(total.detach()))
                    else:
                        # ...le cosine, au RANG de l'epoch.
                        self.scheduler.step()

                lr = float(self.adam_optimizer.param_groups[0]["lr"])
                values = self._terms_to_values(terms, total)

                self._track_best(self._monitored_metric(values), iteration)
                self._record("adam", iteration, values, lr)
                self._update_progress_bar(values)
                self._maybe_log_adam(epoch, n_epochs, values, lr)
                self._maybe_checkpoint(iteration)
                self._adam_epochs_run += 1

                if self._should_early_stop():
                    self._request_stop("early_stopping")
                    break
        finally:
            self._close_progress_bar()

    # ------------------------------------------------------------------
    # Phase 2 — L-BFGS
    # ------------------------------------------------------------------

    def _train_lbfgs(self) -> None:
        """
        Boucle d'affinage L-BFGS avec closure strong_wolfe.

        DEUX SUBTILITÉS À CONNAÎTRE
        ---------------------------
        1. La closure est appelée PLUSIEURS fois par step() (line search).
           On n'enregistre PAS ses valeurs internes : après chaque step,
           on ré-évalue proprement la loss sur les poids courants —
           seule mesure fidèle de l'état du modèle.

        2. Les poids effectifs sont GELÉS (cf. docstring du module) :
           L-BFGS exige un objectif stationnaire, on hérite donc des
           poids du dernier pas Adam.

        GARDE-FOUS INTÉGRÉS
        -------------------
        - try/except autour de step() : une line search qui échoue lève
          un RuntimeError → arrêt propre plutôt que crash ;
        - détection de NaN/inf après chaque pas → restauration du best
          et arrêt (un lr trop grand peut faire diverger les poids).
        """
        self._current_phase = "lbfgs"
        n_iterations = self.train_cfg.lbfgs_iterations

        if self.train_cfg.weighting != "fixed":
            self._log(
                "[info] L-BFGS : poids effectifs gelés aux valeurs finales "
                "de la phase Adam (objectif stationnaire requis)."
            )

        def closure() -> torch.Tensor:
            """
            Closure L-BFGS : zéro les gradients, recalcule la loss,
            backward, RENVOIE la loss (contrat torch.optim.LBFGS).
            """
            self._closure_evals += 1
            self.lbfgs_optimizer.zero_grad(set_to_none=False)
            terms = self._compute_terms()
            total = self._weighted_total(terms)
            total.backward()
            return total

        progress = self._progress(
            range(n_iterations), total=n_iterations, desc="L-BFGS", unit="step"
        )
        try:
            for index in progress:
                iteration = self._global_iteration

                try:
                    self.lbfgs_optimizer.step(closure)
                except RuntimeError as error:
                    self._log(f"[warn] L-BFGS a échoué ({error}) → arrêt du run.")
                    self._request_stop("lbfgs_error")
                    break

                # Mesure fidèle sur les poids courants (cf. docstring) :
                # create_graph=False suffit, on n'entraîne plus ici.
                terms = self._compute_terms(create_graph=False)
                values = self._terms_to_values(terms)

                if not all(math.isfinite(value) for value in values.values()):
                    self._log(
                        "[warn] loss non finie après le pas L-BFGS → divergence "
                        "détectée, restauration du meilleur checkpoint."
                    )
                    self.restore_best_weights()
                    self._request_stop("divergence_nan")
                    break

                lr = float(self.lbfgs_optimizer.param_groups[0]["lr"])
                self._track_best(self._monitored_metric(values), iteration)
                self._record("lbfgs", iteration, values, lr)
                self._update_progress_bar(values)
                self._maybe_log_lbfgs(index, n_iterations, values)
                self._maybe_checkpoint(iteration)
                self._lbfgs_iterations_run += 1

                if self._should_early_stop():
                    self._request_stop("early_stopping")
                    break
        finally:
            self._close_progress_bar()

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def _build_scheduler(self) -> None:
        """
        Construit le scheduler de lr de la phase Adam selon la config.

        POURQUOI ReduceLROnPlateau PAR DÉFAUT ?
            On ne sait pas, a priori, combien d'epochs le problème va
            mettre à converger : un cosine mal dimensionné (T_max trop
            court) écraserait le lr trop tôt, trop long ne ferait rien.
            Le plateau s'adapte à la LOSS observée : « si tu n'améliores
            plus depuis N epochs, divise le lr par `factor` ».
        """
        kind = self.train_cfg.scheduler
        if kind == "plateau":
            self.scheduler: Optional[Any] = ReduceLROnPlateau(
                self.adam_optimizer,
                mode="min",
                factor=self.train_cfg.scheduler_factor,
                patience=self.train_cfg.scheduler_patience,
                min_lr=self.train_cfg.scheduler_min_lr,
            )
        elif kind == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.adam_optimizer,
                T_max=max(self.train_cfg.adam_epochs, 1),
                eta_min=self.train_cfg.scheduler_min_lr,
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------
    # Calcul des losses
    # ------------------------------------------------------------------

    def _compute_terms(self, create_graph: bool = True) -> LossTerms:
        """
        Évalue la loss multi-objectif complète (un seul forward partagé).

        Parameters
        ----------
        create_graph : bool
            True pendant l'entraînement (le résidu doit rester
            différentiable par rapport à θ pour backward) ; False pour
            une évaluation de diagnostic.

        Returns
        -------
        LossTerms
            Termes bruts + pondérés (cf. src.losses).
        """
        return pinn_loss(
            self.model,
            self.ic_batch,
            self.bc_batch,
            self.res_batch,
            self._effective_loss_cfg,
            create_graph=create_graph,
        )

    def _weighted_total(self, terms: LossTerms) -> torch.Tensor:
        """
        Recompose la perte totale avec les poids effectifs COURANTS.

        POURQUOI ne pas utiliser `terms.total` directement ?
            `pinn_loss` a pondéré avec les poids EFFECTIFS au moment du
            forward ; si la pondération dynamique vient de les changer
            (epoch multiple de weighting_update_every), `terms.total` est
            périmé. Recomposer la somme depuis les termes BRUTS est une
            opération scalaire quasi gratuite et toujours à jour.
        """
        weights = self._effective_loss_cfg
        return weights.w_ic * terms.ic + weights.w_bc * terms.bc + weights.w_res * terms.residual

    @staticmethod
    def _terms_to_values(
        terms: LossTerms,
        total: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Extrait les valeurs en floats Python natifs (anti-fuite mémoire :
        stocker des Tenseurs dans l'historique retiendrait les graphes).

        Parameters
        ----------
        terms : LossTerms
            Termes de la loss (bruts).
        total : torch.Tensor, optional
            Perte totale éventuellement recomposée ; défaut terms.total.
        """
        total = terms.total if total is None else total
        return {
            "loss_total": float(total.detach()),
            "loss_ic": float(terms.ic.detach()),
            "loss_bc": float(terms.bc.detach()),
            "loss_res": float(terms.residual.detach()),
        }

    def _monitored_metric(self, values: Dict[str, float]) -> float:
        """Sélectionne la métrique surveillée (total ou résidu)."""
        if self.train_cfg.es_monitor == "residual":
            return values["loss_res"]
        return values["loss_total"]

    # ------------------------------------------------------------------
    # Pondération dynamique
    # ------------------------------------------------------------------

    def _should_update_weights(self, epoch: int) -> bool:
        """
        Décide s'il faut mettre à jour les facteurs adaptatifs à cette epoch.

        Règles : jamais à l'epoch 0 (le premier pas s'effectue toujours
        avec les poids de configuration), pas avant la fin du warmup,
        puis toutes les `weighting_update_every` epochs.
        """
        if self.train_cfg.weighting == "fixed":
            return False
        if epoch == 0 or epoch < self.train_cfg.weighting_warmup:
            return False
        return epoch % self.train_cfg.weighting_update_every == 0

    def _update_adaptive_weights(self, terms: LossTerms) -> None:
        """
        Recalcule les facteurs adaptatifs depuis les termes BRUTS courants.

        Les appels `torch.autograd.grad(..., retain_graph=True)` n'écrivent
        PAS dans `.grad` : le `total.backward()` qui suit reste exact.
        En cas d'échec (graphe dégénéré), on CONSERVE les poids précédents
        plutôt que d'interrompre un run long.
        """
        scheme = self.train_cfg.weighting
        try:
            if scheme == "grad_norm":
                factors = self._factors_grad_norm(terms)
            elif scheme == "lr_annealing":
                factors = self._factors_lr_annealing(terms)
            else:  # pragma: no cover - bloqué par _should_update_weights
                return
        except RuntimeError as error:
            self._log(
                f"[warn] pondération dynamique indisponible cette epoch ({error}) ; "
                "poids précédents conservés."
            )
            return

        if not all(math.isfinite(factor) and factor > 0.0 for factor in factors):
            self._log("[warn] facteurs de pondération non finis ; poids précédents conservés.")
            return

        self._adaptive = factors
        self._refresh_effective_loss_cfg()

    def _factors_grad_norm(self, terms: LossTerms) -> Tuple[float, float, float]:
        """
        Schéma « grad_norm » : w_i ∝ 1 / ‖∇_θ L_i‖₂, normalisé (max = 1).

        PRINCIPE
        --------
        La norme du gradient mesure la FORCE avec laquelle chaque terme
        tire sur θ. À l'initialisation, L_res tire ~100× plus fort que
        L_ic (cf. bilan de l'Étape 2) : le résidu dicte la trajectoire…
        et le résidu seul admet la solution triviale T* ≡ 0. Donner à
        chaque terme un poids inversement proportionnel à la norme de
        son gradient ÉQUILIBRE les tirages (esprit GradNorm, Chen et al.
        2018, transposé aux PINNs).

        NORMALISATION
        -------------
        On divise par le plus grand facteur : les poids restent dans
        (0, 1] — l'échelle de la loss totale ne diverge pas, et l'early
        stopping sur la perte totale garde un sens.

        Notes
        -----
        * Le calcul porte sur TOUS les paramètres entraînables.
        * allow_unused=True : le biais de sortie ne reçoit AUCUN gradient
          du résidu (l'opérateur de la chaleur annule les constantes,
          cf. tests de l'Étape 2) — ce None est traité comme un gradient
          nul.
        * Coût : 3 backward supplémentaires, uniquement aux epochs de
          mise à jour (weighting_update_every).

        Parameters
        ----------
        terms : LossTerms
            Termes BRUTS courants, encore rattachés au graphe.

        Returns
        -------
        (a_ic, a_bc, a_res) : tuple of float
            Facteurs adaptatifs dans (0, 1], max = 1.
        """
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        eps = self.train_cfg.weighting_eps

        norms: List[float] = []
        for term in (terms.ic, terms.bc, terms.residual):
            grads = torch.autograd.grad(
                term, parameters, retain_graph=True, allow_unused=True
            )
            squared_norm = sum(
                float(g.detach().square().sum()) for g in grads if g is not None
            )
            norms.append(math.sqrt(max(squared_norm, 0.0)))

        inverse_norms = [1.0 / max(norm, eps) for norm in norms]
        largest = max(inverse_norms)
        if not math.isfinite(largest) or largest <= 0.0:
            return (1.0, 1.0, 1.0)
        return (
            inverse_norms[0] / largest,
            inverse_norms[1] / largest,
            inverse_norms[2] / largest,
        )

    def _factors_lr_annealing(self, terms: LossTerms) -> Tuple[float, float, float]:
        """
        Schéma « lr_annealing » (Wang, Teng & Perdikaris, ICML 2021).

        PRINCIPE
        --------
        Pour chaque terme, sur les poids de la PREMIÈRE couche (là où
        les échelles d'entrée se mélangent) :

            ŵ_i = max_θ |∇_θ L_i|  /  mean_θ |∇_θ L_i|   ≥ 1

        Ce ratio mesure le CARACTÈRE POINTU du gradient : s'il est grand,
        la mise à jour de L_i est dominée par quelques paramètres —
        signature d'un conditionnement pathologique. L'article anneale
        alors le pas effectif du terme pathologique vers le bas :

            w_i ∝ 1/ŵ_i,  normalisé pour que max(w) = 1.

        POURQUOI LA PREMIÈRE COUCHE SEULEMENT ?
        C'est le choix de l'article : la première couche voit les trois
        entrées (x*, y*, t*) et concentre les pathologies de propagation
        (c'est aussi la couche la plus sensible au gel spectral). Et le
        calcul n'y coûte que quelques milliers d'éléments → quasi gratuit.

        Parameters
        ----------
        terms : LossTerms
            Termes BRUTS courants, encore rattachés au graphe.

        Returns
        -------
        (a_ic, a_bc, a_res) : tuple of float
            Facteurs adaptatifs dans (0, 1], max = 1.
        """
        first_linear = next(
            module for module in self.model.modules() if isinstance(module, nn.Linear)
        )
        parameters = list(first_linear.parameters())
        eps = self.train_cfg.weighting_eps

        ratios: List[float] = []
        for term in (terms.ic, terms.bc, terms.residual):
            grads = torch.autograd.grad(
                term, parameters, retain_graph=True, allow_unused=True
            )
            flattened = torch.cat(
                [g.detach().reshape(-1).abs() for g in grads if g is not None]
            )
            if flattened.numel() == 0:
                ratios.append(1.0)
                continue
            mean_abs = max(float(flattened.mean()), eps)
            max_abs = float(flattened.max())
            ratios.append(max_abs / mean_abs)

        inverse_ratios = [1.0 / max(ratio, eps) for ratio in ratios]
        largest = max(inverse_ratios)
        if not math.isfinite(largest) or largest <= 0.0:
            return (1.0, 1.0, 1.0)
        return (
            inverse_ratios[0] / largest,
            inverse_ratios[1] / largest,
            inverse_ratios[2] / largest,
        )

    def _refresh_effective_loss_cfg(self) -> None:
        """Applique les facteurs adaptatifs : w_eff = w_config × w_adaptatif."""
        base = self.loss_cfg
        a_ic, a_bc, a_res = self._adaptive
        self._effective_loss_cfg = LossConfig(
            w_ic=base.w_ic * a_ic,
            w_bc=base.w_bc * a_bc,
            w_res=base.w_res * a_res,
        )

    # ------------------------------------------------------------------
    # Suivi : record, early stopping, historique, checkpoints
    # ------------------------------------------------------------------

    def _track_best(self, metric: float, iteration: int) -> None:
        """
        Met à jour le suivi du record ; sauvegarde le best si amélioré.

        POURQUOI COPIER LE STATE_DICT SUR CPU ?
            Garder une référence directe aux tenseurs GPU figerait la
            mémoire au moment du record… et pointerait vers des tenseurs
            que Adam réécrit ensuite. Une copie CPU détachée est le
            moyen le plus simple de disposer d'un point de restauration
            (utilisé en cas de divergence L-BFGS) — quelques dizaines de
            Ko pour ~13k paramètres.

        POURQUOI min_delta ?
            Une « amélioration » de 1e-12 sur une loss à 1e-4 n'est
            statistiquement que du bruit d'arrondi float32. Exiger une
            amélioration minimale rend l'early stopping robuste.
        """
        improved = metric < self._best_metric - self.train_cfg.es_min_delta
        if improved:
            self._best_metric = metric
            self._best_epoch = iteration
            self._epochs_without_improvement = 0
            self._best_state_cpu = {
                name: value.detach().to("cpu").clone()
                for name, value in self.model.state_dict().items()
            }
            self._save_checkpoint_file(self.best_checkpoint_path, include_history=False)
        else:
            self._epochs_without_improvement += 1

    def _should_early_stop(self) -> bool:
        """True si l'early stopping est armé ET que la patience est épuisée."""
        return (
            self.train_cfg.early_stopping
            and self._epochs_without_improvement >= self.train_cfg.es_patience
        )

    def _request_stop(self, reason: str) -> None:
        """Enregistre la raison d'arrêt et l'annonce dans les logs."""
        self._stop_reason = reason
        if reason == "early_stopping":
            self._log(
                f"[stop] early stopping : aucune amélioration > "
                f"{self.train_cfg.es_min_delta:.1e} depuis "
                f"{self.train_cfg.es_patience} itérations "
                f"(métrique surveillée : {self.train_cfg.es_monitor})."
            )
        else:
            self._log(f"[stop] arrêt du run : {reason}.")

    def _record(
        self,
        phase: str,
        iteration: int,
        values: Dict[str, float],
        lr: float,
    ) -> None:
        """
        Ajoute une ligne à l'historique + TensorBoard.

        Uniquement des types Python natifs (float/str) : l'historique est
        sérialisable en JSON et ne retient AUCUN graphe autograd.
        """
        weights = self._effective_loss_cfg
        self.history["phase"].append(phase)
        self.history["iteration"].append(iteration)
        self.history["loss_total"].append(values["loss_total"])
        self.history["loss_ic"].append(values["loss_ic"])
        self.history["loss_bc"].append(values["loss_bc"])
        self.history["loss_res"].append(values["loss_res"])
        self.history["w_ic"].append(float(weights.w_ic))
        self.history["w_bc"].append(float(weights.w_bc))
        self.history["w_res"].append(float(weights.w_res))
        self.history["lr"].append(float(lr))
        self._elapsed = time.perf_counter() - self._start_time
        self.history["elapsed"].append(self._elapsed)
        self._global_iteration = iteration + 1

        if self.writer is not None:
            self.writer.add_scalar("loss/total", values["loss_total"], iteration)
            self.writer.add_scalar("loss/ic", values["loss_ic"], iteration)
            self.writer.add_scalar("loss/bc", values["loss_bc"], iteration)
            self.writer.add_scalar("loss/res", values["loss_res"], iteration)
            self.writer.add_scalar("weights/ic", weights.w_ic, iteration)
            self.writer.add_scalar("weights/bc", weights.w_bc, iteration)
            self.writer.add_scalar("weights/res", weights.w_res, iteration)
            self.writer.add_scalar("lr", lr, iteration)

    def _maybe_checkpoint(self, iteration: int) -> None:
        """Écrit last_model.pt périodiquement (protection anti-interruption)."""
        if (iteration + 1) % self.train_cfg.checkpoint_every == 0:
            self._save_checkpoint_file(self.last_checkpoint_path, include_history=True)

    def _save_checkpoint_file(self, path: Path, include_history: bool) -> None:
        """Écrit un checkpoint via src.utils.save_checkpoint (cf. docstring)."""
        save_checkpoint(
            path,
            self.model,
            epoch=self._global_iteration,
            phase=self._current_phase,
            best_loss=self._best_metric if math.isfinite(self._best_metric) else None,
            optimizers={"adam": self.adam_optimizer, "lbfgs": self.lbfgs_optimizer},
            scheduler=self.scheduler,
            history=self.history if include_history else None,
            extra={
                "effective_weights": self._effective_loss_cfg.as_tuple(),
                "adaptive_factors": self._adaptive,
                "weighting": self.train_cfg.weighting,
                "monitor": self.train_cfg.es_monitor,
                "closure_evals": self._closure_evals,
            },
        )

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Affiche un message (via tqdm.write si une barre est active)."""
        if not self.verbose:
            return
        if self._active_bar is not None and _tqdm is not None:
            _tqdm.write(message)
        else:
            print(message)

    def _maybe_log_adam(
        self,
        epoch: int,
        n_epochs: int,
        values: Dict[str, float],
        lr: float,
    ) -> None:
        """Ligne de log Adam : 1 epoch sur log_every, plus première et dernière."""
        if not self.verbose:
            return
        is_last = (epoch + 1) == n_epochs
        if epoch == 0 or is_last or (epoch + 1) % self.train_cfg.log_every == 0:
            weights = self._effective_loss_cfg
            self._log(
                f"[adam {epoch + 1:>5}/{n_epochs}] L={values['loss_total']:.3e} | "
                f"ic={values['loss_ic']:.2e} bc={values['loss_bc']:.2e} "
                f"res={values['loss_res']:.2e} | "
                f"w=({weights.w_ic:.2f},{weights.w_bc:.2f},{weights.w_res:.2f}) | "
                f"lr={lr:.2e} | t+{format_seconds(self._elapsed)}"
            )

    def _maybe_log_lbfgs(
        self,
        index: int,
        n_iterations: int,
        values: Dict[str, float],
    ) -> None:
        """Ligne de log L-BFGS : 1 itération sur lbfgs_log_every + extrêmes."""
        if not self.verbose:
            return
        is_last = (index + 1) == n_iterations
        if index == 0 or is_last or (index + 1) % self.train_cfg.lbfgs_log_every == 0:
            self._log(
                f"[lbfgs {index + 1:>4}/{n_iterations}] L={values['loss_total']:.3e} | "
                f"ic={values['loss_ic']:.2e} bc={values['loss_bc']:.2e} "
                f"res={values['loss_res']:.2e} | "
                f"evals={self._closure_evals} | t+{format_seconds(self._elapsed)}"
            )

    # ------------------------------------------------------------------
    # Barres de progression
    # ------------------------------------------------------------------

    def _progress(
        self,
        iterable: Iterable[int],
        total: int,
        desc: str,
        unit: str,
    ) -> Iterable[int]:
        """
        Enveloppe l'itérable d'une barre tqdm si disponible et demandée.

        Retourne TOUJOURS un itérable : sans tqdm, la boucle tourne
        simplement sans barre (le logging texte prend le relais).
        """
        if self.use_tqdm and _tqdm is not None:
            self._active_bar = _tqdm(
                iterable,
                total=total,
                desc=desc,
                unit=unit,
                dynamic_ncols=True,
                disable=None,  # None → auto-désactivation si la sortie n'est pas un TTY
            )
            return self._active_bar
        self._active_bar = None
        return iterable

    def _update_progress_bar(self, values: Dict[str, float]) -> None:
        """Affiche la loss courante dans le postfixe de la barre tqdm."""
        if self._active_bar is not None:
            self._active_bar.set_postfix_str(f"L={values['loss_total']:.2e}")

    def _close_progress_bar(self) -> None:
        """Ferme proprement la barre tqdm de la phase terminée."""
        if self._active_bar is not None:
            self._active_bar.close()
            self._active_bar = None


__all__ = [
    "PINNTrainer",
    "FitResult",
    "HISTORY_KEYS",
]
