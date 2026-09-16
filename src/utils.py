"""
==============================================================================
src/utils.py — Boîte à outils transverse du projet
==============================================================================

POURQUOI CE MODULE
------------------
Un PINN évalue son résidu PDE sur un nuage de points de collocation.
La QUALITÉ de ce nuage conditionne la qualité de l'apprentissage :

    - Tirage uniforme naïf  → grumeaux, trous (mauvaise couverture).
    - Sobol / LHS           → faible discrépance, couverture homogène.

On fournit aussi des MASQUES géométriques (intérieur, bord, objet chaud)
qui traduisent en booléens NumPy les régions physiques du problème.

Depuis l'Étape 3, le module héberge aussi les utilitaires TRANSVERSES de
la boucle d'entraînement (utilisés par src/trainer.py et main_step3.py) :

    - get_device          : sélection CUDA → MPS → CPU transparente ;
    - save_checkpoint /
      load_checkpoint     : sérialisation torch.save du modèle, des états
                            d'optimiseurs, du scheduler et de l'historique ;
    - plot_loss_history   : figure 2×2 des courbes d'apprentissage
                            (pertes, termes, poids effectifs, lr).

Convention de shapes dans tout le projet
----------------------------------------
    points Sobol/LHS : np.ndarray de shape (n_points, n_dimensions)
    masques          : np.ndarray de shape (n_points,)  dtype=bool
    tenseurs PINN    : torch.Tensor de shape (n_points, 1)
                       ↑ la dimension "1" est le canal scalaire
                         (une seule coordonnée par tenseur : x* OU y* OU t*)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from scipy.stats import qmc


# ==========================================================================
# Reproductibilité
# ==========================================================================

def set_seed(seed: int = 42) -> None:
    """
    Fixe les graines NumPy et PyTorch pour rendre les tirages reproductibles.

    POURQUOI c'est indispensable en PINN / ML :
        Sans graine fixe, chaque run tire des points différents → la loss
        et les figures changent, et on ne peut plus comparer deux expériences
        ni debugger un comportement bizarre.

    Parameters
    ----------
    seed : int
        Entier quelconque (42 par convention communautaire).
        La même graine ⇒ les mêmes séquences pseudo-aléatoires.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Sur GPU, il y a une graine séparée par device
        torch.cuda.manual_seed_all(seed)


# ==========================================================================
# Matériel (device) — Étape 3
# ==========================================================================

def get_device(request: str = "auto") -> torch.device:
    """
    Sélectionne le device PyTorch le plus pertinent disponible.

    POURQUOI ce helper ?
        Le même code doit tourner sans édition manuelle sur un laptop CPU,
        une station CUDA et un Mac Apple Silicon (MPS). Écrire en dur
        `"cuda"` fait planter les deux autres ; écrire `"cpu"` gaspille le
        GPU quand il existe. La résolution automatique ordonne :

            cuda  >  mps  >  cpu

    COMPORTEMENT SELON `request`
    ----------------------------
        "auto" : prend le meilleur device disponible.
        "cpu"  : force le CPU (tests, debug déterministe).
        "cuda" / "mps" : honore la demande SI disponible, sinon affiche un
        avertissement et retombe sur "auto" — plutôt que de lever une
        erreur, pour ne pas bloquer un run long lancé depuis une autre
        machine.

    POURQUOI `torch.backends.mps` VIA getattr ?
        Le backend MPS n'existe pas dans toutes les builds de PyTorch ;
        y accéder directement lèverait AttributeError sur certaines
        installations. L'accès défensif garde le helper portable.

    Parameters
    ----------
    request : {'auto', 'cpu', 'cuda', 'mps'}
        Device demandé.

    Returns
    -------
    torch.device
        Le device effectivement utilisé.

    Raises
    ------
    ValueError
        Si `request` n'est pas l'un des quatre identifiants attendus.
    """
    request = request.lower().strip()
    if request not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError(
            f"Device inconnu : {request!r}. Choisir 'auto', 'cpu', 'cuda' ou 'mps'."
        )

    def _mps_is_available() -> bool:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())

    if request == "cpu":
        return torch.device("cpu")

    if request == "cuda" and not torch.cuda.is_available():
        print("[warn] 'cuda' demandé mais aucun GPU CUDA disponible → fallback 'auto'.")
        request = "auto"
    if request == "mps" and not _mps_is_available():
        print("[warn] 'mps' demandé mais le backend MPS n'est pas disponible → fallback 'auto'.")
        request = "auto"

    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(request)


def format_seconds(seconds: float) -> str:
    """
    Formate une durée en chaîne lisible (« 3.2 s », « 2 min 05 s », « 1 h 04 min »).

    Utilisé par les logs du trainer et le bilan final de main_step3 —
    un chrono lisible est le premier outil pour comparer des run.

    Parameters
    ----------
    seconds : float
        Durée en secondes.

    Returns
    -------
    str
        Durée formatée.
    """
    if seconds < 0.0:
        seconds = 0.0
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        minutes, remainder = divmod(int(seconds), 60)
        return f"{minutes} min {remainder:02d} s"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, remainder = divmod(remainder, 60)
    return f"{hours} h {minutes:02d} min"


# ==========================================================================
# Échantillonnage quasi-aléatoire
# ==========================================================================

def sobol_sample(
    n_points: int,
    n_dimensions: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
    scramble: bool = True,
) -> np.ndarray:
    """
    Tire `n_points` dans [0, 1]^{n_dimensions} via une séquence de Sobol.

    POURQUOI Sobol plutôt qu'un tirage uniforme ?
        Une séquence de Sobol est à FAIBLE DISCRÉPANCE : elle remplit
        l'hypercube plus régulièrement qu'un tirage i.i.d. uniforme.
        Pour un budget de points limité (surtout N_res), cela réduit
        la variance du résidu PDE estimé et accélère la convergence.

    POURQUOI le "scramble" Owen ?
        Le scramble randomise la séquence tout en gardant sa faible
        discrépance. On obtient ainsi plusieurs réalisations indépendantes
        (grâce à `seed`) sans perdre les bonnes propriétés de couverture.

    POURQUOI tirer une puissance de 2 puis tronquer ?
        Les séquences de Sobol sont théoriquement équilibrées sur des
        blocs de taille 2^k. SciPy le recommande ; on tire donc le
        prochain 2^k ≥ n_points, puis on garde les n_points premiers.

    Parameters
    ----------
    n_points : int
        Nombre de points souhaités.
    n_dimensions : int
        Dimension de l'espace (2 pour (x*,y*), 3 pour (x*,y*,t*), …).
    bounds : np.ndarray of shape (n_dimensions, 2), optional
        Bornes (min, max) par dimension. Si None, on reste dans [0, 1]^d.
        Exemple pour scaler le temps vers [0, t*_max] :
            bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, t_star_max]])
    seed : int
        Graine du scramble.
    scramble : bool
        Active le scramble Owen (recommandé : True).

    Returns
    -------
    samples : np.ndarray of shape (n_points, n_dimensions)
        Coordonnées float64 dans le pavé demandé.
    """
    if n_points <= 0:
        return np.zeros((0, n_dimensions), dtype=np.float64)

    # Moteur quasi-Monte-Carlo Sobol (SciPy)
    sobol_engine = qmc.Sobol(
        d=n_dimensions,
        scramble=scramble,
        seed=seed,
    )

    # Prochaine puissance de 2 ≥ n_points (équilibre de la séquence)
    # Ex. n_points=2000 → n_power_of_two = 2048 = 2^11
    n_power_of_two = 1 << int(np.ceil(np.log2(max(n_points, 1))))

    # Tirage puis troncature aux n_points demandés
    # Shape : (n_power_of_two, n_dimensions) → (n_points, n_dimensions)
    unit_samples = sobol_engine.random(n_power_of_two)[:n_points]

    # Scaling optionnel : [0, 1]^d → pavé défini par bounds
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        expected_shape = (n_dimensions, 2)
        if bounds.shape != expected_shape:
            raise ValueError(
                f"`bounds` doit avoir la shape {expected_shape}, "
                f"reçu {bounds.shape}."
            )
        # qmc.scale fait : out = lo + (hi - lo) * unit
        lower_bounds = bounds[:, 0]  # shape (n_dimensions,)
        upper_bounds = bounds[:, 1]  # shape (n_dimensions,)
        unit_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return unit_samples.astype(np.float64)


def latin_hypercube(
    n_points: int,
    n_dimensions: int,
    bounds: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Latin Hypercube Sampling (LHS) dans [0, 1]^{n_dimensions}.

    POURQUOI le LHS ?
        Le LHS garantit que, sur CHAQUE axe, les n_points se répartissent
        en n_points strates équiprobables (une par « ligne » / « colonne »).
        Moins régulier que Sobol en haute dimension, mais excellent en 2D/3D
        et très utilisé en plan d'expériences.

    Parameters
    ----------
    n_points : int
        Nombre de points.
    n_dimensions : int
        Dimension de l'espace.
    bounds : np.ndarray of shape (n_dimensions, 2), optional
        Bornes (min, max) par dimension. None → [0, 1]^d.
    seed : int
        Graine de générateur.

    Returns
    -------
    samples : np.ndarray of shape (n_points, n_dimensions)
    """
    if n_points <= 0:
        return np.zeros((0, n_dimensions), dtype=np.float64)

    lhs_engine = qmc.LatinHypercube(d=n_dimensions, seed=seed)
    unit_samples = lhs_engine.random(n_points)  # shape (n_points, n_dimensions)

    if bounds is not None:
        bounds = np.asarray(bounds, dtype=np.float64)
        expected_shape = (n_dimensions, 2)
        if bounds.shape != expected_shape:
            raise ValueError(
                f"`bounds` doit avoir la shape {expected_shape}, "
                f"reçu {bounds.shape}."
            )
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        unit_samples = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    return unit_samples.astype(np.float64)


# ==========================================================================
# Masques géométriques
# ==========================================================================
# Un "masque" est un tableau de booléens de même longueur que les points.
# True  = le point appartient à la région
# False = le point est hors de la région
# On s'en sert pour :
#   - peindre T*=1 dans l'objet chaud (condition initiale)
#   - vérifier qu'un point BC est bien sur une paroi
#   - filtrer un sous-ensemble pour un plot ou une loss partielle

def mask_interior(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Masque des points STRICTEMENT intérieurs au carré unité [0, 1]².

    POURQUOI "strictement" (ouvert) ?
        Les points de bord sont réservés aux conditions aux limites (BC).
        Le résidu PDE se calcule dans l'OUVERT (0, 1)² pour éviter de
        compter deux fois la même contrainte (BC + résidu sur le même point).

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées, même shape (n_points,) ou broadcastable.
    tolerance : float
        Marge numérique : on exige d'être à plus de `tolerance` du bord.
        Évite qu'un point à 1e-16 du bord soit classé intérieur par erreur
        d'arrondi flottant.

    Returns
    -------
    mask : np.ndarray of bool, same shape as x_star
        True  ↔ point strictement intérieur.
    """
    inside_x = (x_star > tolerance) & (x_star < 1.0 - tolerance)
    inside_y = (y_star > tolerance) & (y_star < 1.0 - tolerance)
    return inside_x & inside_y


def mask_boundary(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Masque des points situés sur le BORD du carré unité [0, 1]².

    Un point est « au bord » si AU MOINS une coordonnée vaut 0 ou 1
    (à `tolerance` près), tout en restant dans le pavé.

    POURQUOI une tolérance plus large (1e-6) que pour l'intérieur ?
        Les points BC sont construits EXACTEMENT à 0 ou 1, mais après
        conversion float32 (PyTorch) la valeur peut dériver légèrement
        (ex. 0.99999994). 1e-6 absorbe cette erreur sans attraper
        d'intérieur légitime.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées.
    tolerance : float
        Tolérance au bord.

    Returns
    -------
    mask : np.ndarray of bool
        True ↔ point sur une paroi (ou un coin).
    """
    on_vertical_wall = (
        (np.abs(x_star - 0.0) <= tolerance) | (np.abs(x_star - 1.0) <= tolerance)
    )
    on_horizontal_wall = (
        (np.abs(y_star - 0.0) <= tolerance) | (np.abs(y_star - 1.0) <= tolerance)
    )
    inside_bounding_box = (
        (x_star >= -tolerance)
        & (x_star <= 1.0 + tolerance)
        & (y_star >= -tolerance)
        & (y_star <= 1.0 + tolerance)
    )
    return inside_bounding_box & (on_vertical_wall | on_horizontal_wall)


def mask_hot_object(
    x_star: np.ndarray,
    y_star: np.ndarray,
    center_x: float = 0.5,
    center_y: float = 0.5,
    radius: float = 0.15,
    shape: str = "disk",
) -> np.ndarray:
    """
    Masque des points situés À L'INTÉRIEUR de l'objet chaud.

    POURQUOI ce masque est central pour l'Étape 1 :
        La condition initiale du problème physique est :
            T*(x*, y*, t*=0) = 1   si (x*, y*) ∈ objet chaud
            T*(x*, y*, t*=0) = 0   sinon
        Ce masque est EXACTEMENT la traduction numérique de cette
        discontinuité. sample_ic() s'en sert pour peindre T*.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées des points à tester.
        Shape typique : (n_points,).
    center_x, center_y : float
        Centre de l'objet en coordonnées adimensionnées (x*, y*).
    radius : float
        - si shape="disk"   : rayon du disque
        - si shape="square" : demi-côté du carré (norme infinie)
    shape : {'disk', 'square'}
        Géométrie de l'objet.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_points,)
        True  → point dans l'objet → on imposera T* = 1
        False → point hors objet  → on imposera T* = 0
    """
    # Vecteur centre → point, composante par composante
    delta_x = np.asarray(x_star, dtype=np.float64) - center_x
    delta_y = np.asarray(y_star, dtype=np.float64) - center_y

    if shape == "disk":
        # Disque : ||(dx, dy)||_2 ≤ radius
        distance_squared = delta_x * delta_x + delta_y * delta_y
        return distance_squared <= radius**2

    if shape == "square":
        # Carré aligné sur les axes : ||(dx, dy)||_∞ ≤ radius
        return (np.abs(delta_x) <= radius) & (np.abs(delta_y) <= radius)

    raise ValueError(
        f"Forme d'objet inconnue : {shape!r}. Choisir 'disk' ou 'square'."
    )


def which_boundary(
    x_star: np.ndarray,
    y_star: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Identifie LA paroi d'appartenance de chaque point.

    Codage entier (utile pour colorer un plot ou pondérer une loss) :
        0 = paroi gauche   (x* = 0)
        1 = paroi droite   (x* = 1)
        2 = paroi bas      (y* = 0)
        3 = paroi haut     (y* = 1)
       -1 = pas un point de bord

    POURQUOI une priorité x > y sur les coins ?
        Un coin appartient à DEUX parois. Pour éviter un double comptage
        dans les histos / losses, on le rattache arbitrairement à la
        paroi verticale (x). Le choix est conventionnel, l'important
        est qu'il soit déterministe.

    Parameters
    ----------
    x_star, y_star : np.ndarray
        Coordonnées adimensionnées.
    tolerance : float
        Tolérance au bord.

    Returns
    -------
    wall_labels : np.ndarray of int64, same shape as x_star
    """
    # Initialise tout à "non-bord"
    wall_labels = np.full(np.broadcast(x_star, y_star).shape, -1, dtype=np.int64)

    # Parois verticales en premier (priorité coins)
    wall_labels = np.where(np.abs(x_star - 0.0) <= tolerance, 0, wall_labels)
    wall_labels = np.where(np.abs(x_star - 1.0) <= tolerance, 1, wall_labels)

    # Parois horizontales seulement si pas déjà tagué vertical
    not_on_vertical = wall_labels < 0
    wall_labels = np.where(
        not_on_vertical & (np.abs(y_star - 0.0) <= tolerance), 2, wall_labels
    )
    wall_labels = np.where(
        not_on_vertical & (np.abs(y_star - 1.0) <= tolerance), 3, wall_labels
    )
    return wall_labels


# ==========================================================================
# Conversion NumPy → PyTorch
# ==========================================================================

def to_tensor(
    array: np.ndarray,
    requires_grad: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convertit un tableau NumPy en tenseur PyTorch colonne.

    POURQUOI shape (N, 1) et pas (N,) ?
        PyTorch et les MLP attendent en général un tenseur 2D
        (batch_size, n_features). Ici chaque coordonnée (x*, y* ou t*)
        est une "feature" scalaire → n_features = 1.
        Shape finale : (n_points, 1).

    POURQUOI requires_grad=True sur les COORDONNÉES ?
        C'est LE point clé d'un PINN. Pour calculer le résidu
            r = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)
        on a besoin des dérivées de la sortie du réseau par rapport
        à ses ENTRÉES (x*, y*, t*). Autograd ne les construira que si
        ces entrées ont requires_grad=True.

        En revanche la CIBLE T* (IC/BC) a requires_grad=False : c'est
        une constante physique, pas une variable d'optimisation.

    Parameters
    ----------
    array : np.ndarray
        Données source, shape (N,) ou (N, 1) ou (N, C).
    requires_grad : bool
        Active le suivi autograd (True pour x*, y*, t* ; False pour T*).
    device : str
        "cpu" ou "cuda".
    dtype : torch.dtype
        float32 par défaut (bon compromis précision / vitesse / VRAM).

    Returns
    -------
    tensor : torch.Tensor
        Shape (N, 1) si l'entrée était 1-D, sinon shape d'origine.
        dtype et device conformes aux arguments.
    """
    tensor = torch.as_tensor(array, dtype=dtype, device=device)

    # Si on reçoit un vecteur 1-D (N,), on l'amène en colonne (N, 1)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)  # (N,) → (N, 1)

    # Active (ou non) le graphe de calcul autograd sur ce tenseur
    tensor = tensor.requires_grad_(requires_grad)
    return tensor


def unit_cube_bounds(n_dimensions: int) -> np.ndarray:
    """
    Construit le tableau de bornes [[0, 1], [0, 1], …] pour n_dimensions.

    Utile pour appeler sobol_sample / latin_hypercube avec un pavé unité
    explicite (plutôt que bounds=None), ce qui rend l'intention lisible.

    Parameters
    ----------
    n_dimensions : int
        Nombre de dimensions.

    Returns
    -------
    bounds : np.ndarray of shape (n_dimensions, 2)
        Chaque ligne vaut [0.0, 1.0].
    """
    return np.tile(np.array([[0.0, 1.0]]), (n_dimensions, 1))


# ==========================================================================
# Checkpoints — Étape 3
# ==========================================================================

def save_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    *,
    epoch: Optional[int] = None,
    phase: Optional[str] = None,
    best_loss: Optional[float] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    scheduler: Optional[Any] = None,
    history: Optional[Dict[str, list]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Sauvegarde un checkpoint complet et autoportant.

    POURQUOI AUTANT DE CHAMPS ?
        Un checkpoint qui ne contient QUE les poids (state_dict) permet de
        recharger le modèle pour l'INFÉRENCE, mais pas de comprendre le
        run ni de le poursuivre. On sérialise donc aussi :

          - les états des OPTIMISEURS (momenta d'Adam, paires (s, y) de
            L-BFGS) : indispensables pour reprendre l'optimisation là où
            elle s'était arrêtée sans casser la mémoire de courbure ;
          - l'état du SCHEDULER de lr (patience accumulée, lr courant) ;
          - l'HISTORIQUE complet des pertes (pour re-tracer les courbes
            ou diagnostiquer a posteriori) ;
          - un dictionnaire `extra` libre (poids effectifs, graine, …).

    POURQUOI COPIER LE state_dict SUR CPU ?
        Sur GPU, les tenseurs du state_dict vivent en VRAM ; un
        checkpoint écrit depuis CUDA serait alors plus difficile à
        recharger sur une machine CPU (il faudrait map_location).
        Copier sur CPU avant torch.save rend le fichier PORTABLE.

    Parameters
    ----------
    path : str or Path
        Chemin du fichier (ex. checkpoints/best_model.pt). Les parents
        sont créés au besoin.
    model : torch.nn.Module
        Modèle à sérialiser (ses bornes de normalisation, enregistrées
        comme buffers, suivent automatiquement le state_dict).
    epoch, phase : optional
        Itération globale et phase ("adam" / "lbfgs") au moment de la sauvegarde.
    best_loss : float, optional
        Meilleure valeur de la métrique surveillée.
    optimizers : dict[str, torch.optim.Optimizer], optional
        Optimiseurs nommés (ex. {"adam": …, "lbfgs": …}) dont on veut
        préserver l'état.
    scheduler : optional
        Scheduler de lr ( ReduceLROnPlateau, CosineAnnealingLR, …).
    history : dict[str, list], optional
        Historique d'entraînement (clés = loss_total, loss_ic, …).
    extra : dict[str, Any], optional
        Métadonnées supplémentaires (poids effectifs, schéma de
        pondération, graine…).

    Returns
    -------
    Path
        Le chemin effectivement écrit (résolu en chemin absolu).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Copie CPU du state_dict → checkpoint portable d'une machine à l'autre
    model_state = {
        name: value.detach().to("cpu").clone()
        for name, value in model.state_dict().items()
    }

    payload: Dict[str, Any] = {
        "format_version": 1,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_class": type(model).__name__,
        "epoch": epoch,
        "phase": phase,
        "best_loss": best_loss,
        "model_state_dict": model_state,
        "optimizer_state_dicts": {
            name: optimizer.state_dict()
            for name, optimizer in (optimizers or {}).items()
        },
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "extra": extra or {},
    }

    torch.save(payload, path)
    return path.resolve()


def load_checkpoint(
    path: Union[str, Path],
    *,
    model: Optional[torch.nn.Module] = None,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    scheduler: Optional[Any] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Recharge un checkpoint produit par `save_checkpoint`.

    POURQUOI `weights_only=False` ?
        Le payload contient des objets standards de PyTorch (state_dicts,
        dictionnaires, listes, floats) produits PAR NOTRE PROPRE trainer :
        il n'y a rien à désérialiser de non fiable. Le mode strict
        `weights_only=True` de torch.load (défaut depuis PyTorch 2.6)
        refuse certains conteneurs d'états d'optimiseurs ; on lève donc
        explicitement cette restriction pour nos fichiers, en documentant
        qu'il ne faut PAS charger ainsi un checkpoint de provenance inconnue.

    POURQUOI `map_location="cpu"` PAR DÉFAUT ?
        Un checkpoint sauvé depuis le GPU contient des tenseurs estampillés
        "cuda:0". Le chargement par défaut essaierait de les remettre en
        VRAM et échouerait sur une machine sans GPU. Tout atterrir sur CPU
        puis déplacer le modèle ensuite (model.to(device)) est toujours sûr.

    Parameters
    ----------
    path : str or Path
        Chemin du fichier checkpoint.
    model : torch.nn.Module, optional
        Si fourni, ses poids sont remplacés par ceux du checkpoint.
        ATTENTION : le modèle doit être reconstruit avec la MÊME
        architecture (cf. PINN.from_config) — sinon load_state_dict lève.
    optimizers : dict[str, torch.optim.Optimizer], optional
        Optimiseurs nommés dont on restaure l'état (même naming qu'à la
        sauvegarde, ex. {"adam": …}).
    scheduler : optional
        Scheduler dont on restaure l'état.
    map_location : str or torch.device
        Device cible des tenseurs chargés ("cpu" par défaut, cf. ci-dessus).

    Returns
    -------
    payload : dict[str, Any]
        Le checkpoint complet (clés epoch, phase, best_loss, history,
        extra, …) — utile pour afficher un bilan ou rebrancher un run.

    Raises
    ------
    FileNotFoundError
        Si `path` n'existe pas.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path.resolve()}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        state_dict = payload.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                f"Le checkpoint {path.name} ne contient pas de 'model_state_dict'."
            )
        model.load_state_dict(state_dict)

    if optimizers:
        saved_optimizers = payload.get("optimizer_state_dicts", {})
        for name, optimizer in optimizers.items():
            if name in saved_optimizers:
                optimizer.load_state_dict(saved_optimizers[name])

    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])

    return payload


# ==========================================================================
# Figures d'apprentissage — Étape 3
# ==========================================================================

def plot_loss_history(
    history: Dict[str, list],
    out_path: Union[str, Path],
    *,
    title: str = "PINN — courbes d'apprentissage",
    best_epoch: Optional[int] = None,
    early_stopped: bool = False,
) -> Path:
    """
    Trace la figure 2×2 des courbes d'apprentissage et la sauvegarde en PNG.

    Layout
    ------
        (0, 0) Perte totale (échelle log) + marqueur du best + ligne de
               bascule Adam → L-BFGS
        (0, 1) Les trois termes BRUTS L_ic, L_bc, L_res (échelle log) —
               c'est ici que se lit le déséquilibre (cf. Étape 2)
        (1, 0) Poids EFFECTIFS w_ic, w_bc, w_res (schéma dynamique ou plat)
        (1, 1) Learning rate (échelle log) — visualise le scheduler

    POURQUOI L'ÉCHELLE LOG PARTOUT ?
        Une loss PINN descend de ~10 (initialisation) à ~1e-6 (convergence
        L-BFGS) : six ordres de grandeur. En échelle linéaire, tout se
        passerait dans les 50 premières itérations et le reste serait
        visuellement plat — exactement ce qu'on veut éviter de conclure.

    POURQUOI UN PLANCHER À 1e-16 ?
        matplotlib ignore silencieusement les valeurs ≤ 0 en échelle log ;
        une loss exactement nulle (BC parfaitement satisfaites en fin de
        run, arrondi float32) ferait disparaître la courbe. Le plancher
        garde les lignes visibles sans déformer la lecture.

    POURQUOI L'IMPORT MATPLOTLIB EST-IL PARESSEUX ?
        src.utils est importé par TOUS les modules du projet (y compris
        les tests) ; matplotlib coûte ~0,5 s à importer. Ne le charger
        que lorsqu'on trace réellement allège chaque `import src`.

    Parameters
    ----------
    history : dict[str, list]
        Historique produit par PINNTrainer (clés phase, iteration,
        loss_total, loss_ic, loss_bc, loss_res, w_ic, w_bc, w_res, lr).
    out_path : str or Path
        Chemin du PNG de sortie (ex. step3_training_losses.png).
    title : str
        Titre principal de la figure.
    best_epoch : int, optional
        Itération du meilleur checkpoint (marqueur sur la perte totale).
    early_stopped : bool
        Si True, le titre le mentionne (le run s'est arrêté avant la fin
        du planning — information cruciale à la relecture).

    Returns
    -------
    Path
        Chemin résolu du PNG écrit.
    """
    import matplotlib.pyplot as plt

    required_keys = ("iteration", "loss_total", "loss_ic", "loss_bc", "loss_res")
    missing = [key for key in required_keys if key not in history]
    if missing:
        raise KeyError(f"L'historique ne contient pas les clés : {missing}.")
    if len(history["iteration"]) == 0:
        raise ValueError("Historique vide : rien à tracer.")

    iterations = np.asarray(history["iteration"], dtype=float)
    phases = list(history.get("phase", []))
    n_adam = sum(1 for phase_name in phases if phase_name == "adam")

    def _positive(values: list) -> np.ndarray:
        """Clamp à un plancher > 0 pour l'échelle log (cf. docstring)."""
        return np.maximum(np.asarray(values, dtype=float), 1e-16)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    subtitle = "arrêt anticipé (early stopping)" if early_stopped else "planning complet exécuté"
    figure.suptitle(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")

    def _mark_phase_boundary(axis: Any) -> None:
        """Ligne verticale à la bascule Adam → L-BFGS, si elle existe."""
        if 0 < n_adam < len(iterations):
            axis.axvline(
                n_adam - 0.5, color="#3D405B", linestyle="--", linewidth=1.2,
                label="bascule Adam → L-BFGS",
            )
            axis.legend(fontsize=8, loc="best")

    # ---- (0, 0) Perte totale -------------------------------------------
    ax_total: Any = axes[0, 0]
    ax_total.plot(iterations, _positive(history["loss_total"]), color="#4C72B0", linewidth=1.8)
    ax_total.set_yscale("log")
    _mark_phase_boundary(ax_total)
    if best_epoch is not None and best_epoch in set(history["iteration"]):
        best_index = list(history["iteration"]).index(best_epoch)
        best_value = _positive(history["loss_total"])[best_index]
        ax_total.scatter([best_epoch], [best_value], color="#C44E52", zorder=5, s=45)
        ax_total.annotate(
            f"best {best_value:.2e}",
            (best_epoch, best_value),
            textcoords="offset points", xytext=(8, 6), fontsize=8, color="#C44E52",
        )
    ax_total.set_xlabel("itération")
    ax_total.set_ylabel(r"$\mathcal{L}_{total}$")
    ax_total.set_title("Perte totale pondérée")
    ax_total.grid(True, alpha=0.3)

    # ---- (0, 1) Termes individuels --------------------------------------
    ax_terms: Any = axes[0, 1]
    term_specs = (
        ("loss_ic", r"$\mathcal{L}_{ic}$", "#55A868"),
        ("loss_bc", r"$\mathcal{L}_{bc}$", "#F58518"),
        ("loss_res", r"$\mathcal{L}_{res}$", "#8172B3"),
    )
    for key, label, color in term_specs:
        ax_terms.plot(iterations, _positive(history[key]), label=label, color=color, linewidth=1.4)
    ax_terms.set_yscale("log")
    _mark_phase_boundary(ax_terms)
    ax_terms.set_xlabel("itération")
    ax_terms.set_ylabel("terme brut (MSE)")
    ax_terms.set_title("Termes de la loss (bruts, non pondérés)")
    ax_terms.legend(fontsize=9)
    ax_terms.grid(True, alpha=0.3)

    # ---- (1, 0) Poids effectifs ------------------------------------------
    ax_weights: Any = axes[1, 0]
    if all(key in history for key in ("w_ic", "w_bc", "w_res")):
        weight_specs = (
            ("w_ic", r"$w_{ic}$", "#55A868"),
            ("w_bc", r"$w_{bc}$", "#F58518"),
            ("w_res", r"$w_{res}$", "#8172B3"),
        )
        for key, label, color in weight_specs:
            ax_weights.plot(iterations, np.asarray(history[key], dtype=float),
                            label=label, color=color, linewidth=1.4)
        _mark_phase_boundary(ax_weights)
        ax_weights.set_xlabel("itération")
        ax_weights.set_ylabel("poids effectif")
        ax_weights.set_title("Pondération effective (config × adaptatif)")
        ax_weights.legend(fontsize=9)
        ax_weights.grid(True, alpha=0.3)
    else:
        ax_weights.axis("off")

    # ---- (1, 1) Learning rate --------------------------------------------
    ax_lr: Any = axes[1, 1]
    if "lr" in history:
        ax_lr.plot(iterations, _positive(history["lr"]), color="#CCB974", linewidth=1.6)
        ax_lr.set_yscale("log")
        _mark_phase_boundary(ax_lr)
        ax_lr.set_xlabel("itération")
        ax_lr.set_ylabel("learning rate")
        ax_lr.set_title("Learning rate (scheduler)")
        ax_lr.grid(True, alpha=0.3)
    else:
        ax_lr.axis("off")

    figure.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return out_path.resolve()


__all__ = [
    "set_seed",
    "get_device",
    "format_seconds",
    "sobol_sample",
    "latin_hypercube",
    "mask_interior",
    "mask_boundary",
    "mask_hot_object",
    "which_boundary",
    "to_tensor",
    "unit_cube_bounds",
    "save_checkpoint",
    "load_checkpoint",
    "plot_loss_history",
]
