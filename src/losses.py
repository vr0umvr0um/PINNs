"""
==============================================================================
src/losses.py — Fonction de perte multi-objectif du PINN
==============================================================================

RÔLE DE CE MODULE (Étape 2)
---------------------------
Le sujet demande explicitement (équation 4) :

    L(θ)  =  w_ic · L_ic  +  w_bc · L_bc  +  w_res · L_res

Trois contraintes, trois termes, trois familles de points (celles de
l'Étape 1). Chacun répond à une question différente :

    L_ic   « à t* = 0, retrouves-tu l'objet chaud ? »
           MSE( T_θ(x*, y*, 0) , T*_cible )      sur les N_ic points IC

    L_bc   « les murs restent-ils à la température ambiante ? »
           MSE( T_θ(paroi, t*) , 0 )             sur les N_bc points BC

    L_res  « ton champ obéit-il à l'équation de la chaleur ? »
           MSE( ∂T*/∂t* − Δ*T* , 0 )             sur les N_res points

POURQUOI CETTE DÉCOMPOSITION EST L'IDÉE CENTRALE DU PINN
---------------------------------------------------------
L_ic et L_bc sont des termes SUPERVISÉS classiques : on connaît la
réponse et on la compare. L_res, lui, n'a AUCUNE donnée cible : sa
« vérité terrain » est l'équation de la physique. C'est ce terme qui
permet d'apprendre la solution partout dans le domaine sans jamais avoir
mesuré la moindre température à l'intérieur de la pièce.

POURQUOI LES TROIS TERMES SONT INDISPENSABLES ENSEMBLE
-------------------------------------------------------
Retirer un terme casse le problème de façon instructive :

    sans L_res  → simple régression sur le bord et l'instant initial ;
                  le réseau interpole n'importe comment entre les deux.
    sans L_ic   → T* ≡ 0 devient une solution PARFAITE (elle vérifie
                  l'équation et les BC). Le réseau converge vers le
                  champ nul : le piège le plus courant en PINN.
    sans L_bc   → la chaleur n'est plus évacuée par les murs ; le champ
                  dérive vers une solution non physique.

POURQUOI LA MSE ET PAS UNE AUTRE NORME
---------------------------------------
La MSE correspond à une norme L² discrète sur les points de collocation.
C'est la norme naturelle pour une EDP parabolique (l'énergie du système
est une intégrale quadratique), elle est lisse partout — donc bien
adaptée aux optimiseurs de second ordre comme L-BFGS prévu à l'Étape 3 —
et elle pénalise fortement les quelques points aberrants, typiquement
ceux situés sur le bord de l'objet chaud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from config import DEFAULT_LOSS_CONFIG, LossConfig
from src.physics import TemperatureField, pde_residual

# Un batch de collocation = dict nom → tenseur (sortie de src.sampling).
CollocationBatch = Dict[str, torch.Tensor]


# ==========================================================================
# Conteneur de résultats
# ==========================================================================

@dataclass
class LossTerms:
    """
    Décomposition complète de la loss, pour le logging et le diagnostic.

    POURQUOI RENVOYER AUTANT QUE ÇA plutôt qu'un seul scalaire ?
        Pendant l'entraînement (Étape 3), la question n'est jamais
        « la loss descend-elle ? » mais « QUEL terme bloque ? ».
        Une loss totale qui stagne à 1e-2 peut cacher :
            - L_ic = 1e-2 et L_res = 1e-6  → l'IC raide résiste ;
            - L_res = 1e-2 et L_ic = 1e-6  → l'EDP n'est pas respectée.
        Le diagnostic et le remède sont opposés. On conserve donc les
        termes BRUTS (comparables entre eux, indépendants des poids) ET
        les termes PONDÉRÉS (qui montrent la contribution réelle au
        gradient).

    Attributs
    ---------
    total : Tensor scalaire
        La quantité à minimiser : c'est sur elle qu'on appelle backward().
        Seul terme rattaché au graphe autograd de bout en bout.
    ic, bc, residual : Tensor scalaires
        Termes BRUTS, non pondérés (MSE pure).
    weighted_ic, weighted_bc, weighted_residual : Tensor scalaires
        Termes multipliés par leur poids — leur somme vaut `total`.
    """

    total: torch.Tensor
    ic: torch.Tensor
    bc: torch.Tensor
    residual: torch.Tensor
    weighted_ic: torch.Tensor
    weighted_bc: torch.Tensor
    weighted_residual: torch.Tensor

    def as_floats(self) -> Dict[str, float]:
        """
        Convertit tous les termes en float Python natifs.

        POURQUOI `.detach()` implicite via float() ?
            Stocker des Tensors dans un historique d'entraînement
            retiendrait tout le graphe autograd en mémoire à chaque
            epoch → fuite mémoire garantie sur quelques milliers
            d'itérations. On extrait des floats pour l'historique.
        """
        return {
            "total": float(self.total.detach()),
            "ic": float(self.ic.detach()),
            "bc": float(self.bc.detach()),
            "residual": float(self.residual.detach()),
            "weighted_ic": float(self.weighted_ic.detach()),
            "weighted_bc": float(self.weighted_bc.detach()),
            "weighted_residual": float(self.weighted_residual.detach()),
        }

    def __str__(self) -> str:
        """Ligne compacte de log, lisible dans un terminal."""
        values = self.as_floats()
        return (
            f"L={values['total']:.4e}  |  "
            f"L_ic={values['ic']:.4e}  "
            f"L_bc={values['bc']:.4e}  "
            f"L_res={values['residual']:.4e}"
        )


# ==========================================================================
# Termes individuels
# ==========================================================================

def loss_initial(
    model: TemperatureField,
    ic_batch: CollocationBatch,
) -> torch.Tensor:
    """
    Terme de CONDITION INITIALE.

        L_ic = (1/N_ic) Σ ( T_θ(x*, y*, 0) − T*_cible )²

    avec T*_cible = 1 dans l'objet chaud, 0 ailleurs.

    POURQUOI C'EST LE TERME LE PLUS DIFFICILE
        La cible est un CRÉNEAU : elle saute de 0 à 1 sur le bord de
        l'objet. Un MLP à activations lisses ne peut pas reproduire une
        discontinuité exacte ; il l'approche par une transition raide et
        oscille légèrement autour (phénomène de Gibbs). C'est
        précisément la « raideur de la condition initiale » que l'Étape 3
        devra traiter (pondération dynamique, resampling adaptatif ou
        hard-constraints).

    Le batch IC porte t* = 0 pour tous ses points : on n'a donc PAS
    besoin de le forcer ici, l'Étape 1 le garantit déjà (et le test
    `validate_ic` le verrouille).

    Parameters
    ----------
    model : TemperatureField
        Réseau T_θ (ou tout callable compatible).
    ic_batch : dict[str, Tensor]
        Sortie de `sample_ic` : clés x_star, y_star, t_star, T_star.

    Returns
    -------
    torch.Tensor
        Scalaire (shape ()), rattaché au graphe autograd.
    """
    T_pred = model(
        ic_batch["x_star"],
        ic_batch["y_star"],
        ic_batch["t_star"],
    )  # (N_ic, 1)

    return F.mse_loss(T_pred, ic_batch["T_star"])


def loss_boundary(
    model: TemperatureField,
    bc_batch: CollocationBatch,
) -> torch.Tensor:
    """
    Terme de CONDITIONS AUX LIMITES (Dirichlet homogène).

        L_bc = (1/N_bc) Σ ( T_θ(paroi, t*) − 0 )²

    Physique : les 4 murs sont maintenus à T_amb pour tout t, ce qui
    donne T* = 0 sur ∂Ω après adimensionnement.

    POURQUOI COMPARER À `bc_batch["T_star"]` PLUTÔT QU'À ZÉRO EN DUR ?
        Le batch transporte déjà sa cible (nulle par défaut, mais
        `sample_bc` accepte un paramètre `T_star_bc`). Lire la cible
        depuis le batch rend le code réutilisable tel quel si l'on veut
        plus tard des murs chauffés (Dirichlet non homogène), sans
        toucher à la loss.

    Returns
    -------
    torch.Tensor
        Scalaire rattaché au graphe autograd.
    """
    T_pred = model(
        bc_batch["x_star"],
        bc_batch["y_star"],
        bc_batch["t_star"],
    )  # (N_bc, 1)

    return F.mse_loss(T_pred, bc_batch["T_star"])


def loss_residual(
    model: TemperatureField,
    res_batch: CollocationBatch,
    create_graph: bool = True,
) -> torch.Tensor:
    """
    Terme de RÉSIDU PDE — le terme « physique » du PINN.

        L_res = (1/N_res) Σ ( ∂T*/∂t* − Δ*T* )²

    POURQUOI IL N'Y A AUCUNE DONNÉE CIBLE ICI
        La « vérité » n'est pas un tableau de mesures mais l'équation
        elle-même. On compare le résidu à zéro, et zéro est un tenseur
        constant : aucune information extérieure n'entre dans ce terme.
        C'est ce qui permet au PINN d'apprendre la solution partout dans
        le domaine sans capteur à l'intérieur de la pièce.

    Parameters
    ----------
    model : TemperatureField
        Réseau T_θ.
    res_batch : dict[str, Tensor]
        Sortie de `sample_residual` : clés x_star, y_star, t_star
        (pas de T_star — c'est normal, il n'y a pas de cible).
    create_graph : bool
        True pendant l'entraînement pour que la loss soit
        différentiable par rapport à θ. False pour un diagnostic.

    Returns
    -------
    torch.Tensor
        Scalaire rattaché au graphe autograd (si create_graph=True).
    """
    residual = pde_residual(
        model,
        res_batch["x_star"],
        res_batch["y_star"],
        res_batch["t_star"],
        create_graph=create_graph,
    )  # (N_res, 1)

    # MSE contre la cible constante 0 : on prend directement la moyenne
    # des carrés, sans allouer un tenseur de zéros.
    return torch.mean(residual**2)


# ==========================================================================
# Loss totale
# ==========================================================================

def pinn_loss(
    model: TemperatureField,
    ic_batch: CollocationBatch,
    bc_batch: CollocationBatch,
    res_batch: CollocationBatch,
    loss_cfg: Optional[LossConfig] = None,
    create_graph: bool = True,
) -> LossTerms:
    """
    Loss multi-objectif complète du PINN (équation 4 du sujet).

        L(θ) = w_ic · L_ic  +  w_bc · L_bc  +  w_res · L_res

    C'est la fonction que l'Étape 3 passera à Adam puis à L-BFGS.

    POURQUOI UNE SEULE SOMME PONDÉRÉE PLUTÔT QUE TROIS BACKWARD ?
        Les trois termes partagent les MÊMES paramètres θ. Une unique
        somme produit un unique graphe, donc un seul `backward()` :
        PyTorch accumule automatiquement ∂L/∂θ = Σ w_i ∂L_i/∂θ. Faire
        trois backward séparés donnerait le même gradient mais
        parcourrait le réseau trois fois — plus lent, sans bénéfice.

    Parameters
    ----------
    model : TemperatureField
        Réseau T_θ à évaluer.
    ic_batch, bc_batch, res_batch : dict[str, Tensor]
        Les trois familles de points produites à l'Étape 1.
    loss_cfg : LossConfig, optional
        Pondérations (w_ic, w_bc, w_res). None → DEFAULT_LOSS_CONFIG.
    create_graph : bool
        Transmis au terme résidu (True pour entraîner).

    Returns
    -------
    LossTerms
        `.total` est le scalaire à minimiser ; les autres champs
        servent au logging et au diagnostic.
    """
    loss_cfg = loss_cfg or DEFAULT_LOSS_CONFIG

    # --- Termes bruts (non pondérés) -----------------------------------
    raw_ic = loss_initial(model, ic_batch)
    raw_bc = loss_boundary(model, bc_batch)
    raw_residual = loss_residual(model, res_batch, create_graph=create_graph)

    # --- Application des poids -----------------------------------------
    weighted_ic = loss_cfg.w_ic * raw_ic
    weighted_bc = loss_cfg.w_bc * raw_bc
    weighted_residual = loss_cfg.w_res * raw_residual

    total = weighted_ic + weighted_bc + weighted_residual

    return LossTerms(
        total=total,
        ic=raw_ic,
        bc=raw_bc,
        residual=raw_residual,
        weighted_ic=weighted_ic,
        weighted_bc=weighted_bc,
        weighted_residual=weighted_residual,
    )


__all__ = [
    "LossTerms",
    "loss_initial",
    "loss_boundary",
    "loss_residual",
    "pinn_loss",
]
