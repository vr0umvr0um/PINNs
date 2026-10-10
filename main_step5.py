#!/usr/bin/env python3
"""
==============================================================================
main_step5.py — Étape 5 : Démonstrateur interactif Gradio
==============================================================================

Lance l'interface qui permet de :
    - faire varier l'objet chaud (forme, centre, rayon) et la physique
      (α, T_amb, T_obj) ;
    - visualiser la carte de chaleur PRÉDITE, avec un slider temporel ;
    - comparer à la solution de référence (différences finies) et afficher
      la carte d'ERREUR, la coupe et l'erreur L2 relative au cours du temps ;
    - ré-entraîner le PINN pour un nouvel objet.

Usage
-----
    python main_step3.py                  # (une fois) produit checkpoints/best_model.pt
    python main_step5.py
    python main_step5.py --checkpoint checkpoints/last_model.pt --port 7861
    python main_step5.py --train-if-missing   # entraîne un réseau rapide si aucun checkpoint
    python main_step5.py --share              # lien public temporaire Gradio
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from config import DataConfig
from src.inference import load_trained_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Étape 5 — démonstrateur Gradio")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt",
                        help="Checkpoint du PINN (défaut : checkpoints/best_model.pt)")
    parser.add_argument("--activation", default="tanh",
                        help="Activation à supposer pour un ancien checkpoint sans architecture")
    parser.add_argument("--train-if-missing", action="store_true",
                        help="Si le checkpoint n'existe pas, entraîne un réseau rapide (~1 min)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Lien public temporaire Gradio")
    parser.add_argument("--n-grid", type=int, default=81, help="Résolution de la grille de comparaison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from src.gradio_app import build_app
    except ImportError as error:
        print(f"[ERREUR] Gradio indisponible ({error}). Installer : pip install gradio")
        return 1

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_file():
        model, trained_cfg, payload = load_trained_model(
            checkpoint, fallback_activation=args.activation
        )
        if trained_cfg is None:
            print("[info] Checkpoint sans DataConfig : objet par défaut supposé.")
            trained_cfg = DataConfig()
        source = str(checkpoint)
        print(f"[OK] PINN chargé depuis {checkpoint} (itération {payload.get('epoch')}).")
    elif args.train_if_missing:
        from src.demo import retrain_for_object

        print(f"[info] {checkpoint} introuvable → entraînement rapide (~1 min)…")
        result = retrain_for_object(
            DataConfig(),
            progress=lambda fraction, message: print(f"  {100 * fraction:3.0f} %  {message}", end="\r"),
            checkpoint_dir=tempfile.mkdtemp(prefix="pinn_step5_"),
        )
        print()
        model, trained_cfg, source = result.model, result.data_cfg, "entraînement rapide (non sauvegardé)"
    else:
        print(f"[ERREUR] Checkpoint introuvable : {checkpoint.resolve()}\n"
              "  → Lancer d'abord `python main_step3.py`, ou ajouter --train-if-missing.")
        return 1

    app = build_app(model, trained_cfg, n_grid=args.n_grid, source=source)
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    sys.exit(main())
