# Instructions pour les agents — projet PINN thermique 2D

Projet académique (M2) : prédire la diffusion thermique 2D instationnaire par un
PINN (PyTorch), le valider contre un solveur aux différences finies, et le montrer
dans un démonstrateur Gradio. Le sujet est dans `PINNs_M2_Temperature_prediction.pdf`.
**Livrables du sujet** : dépôt Git propre et documenté, présentation de 5 minutes,
démonstrateur interactif. Lire `README.md` avant toute modification (il décrit
chaque étape, les formules et les résultats mesurés).

## Numérotation des étapes (piège déjà rencontré)

La numérotation du **sujet** fait foi. Le README avait été faux à ce sujet et a été corrigé.

| Étape | Contenu | État |
|-------|---------|------|
| 1 | Adimensionnement + échantillonnage | ✅ |
| 2 | Architecture, autograd, loss | ✅ |
| 3 | Entraînement Adam → L-BFGS + technique pour l'IC raide | ✅ |
| 4 | **Validation numérique** (solveur DF, MSE, L2 relative) | ✅ |
| 5 | **Démonstrateur Gradio** | ✅ |
| 6 | **Présentation de 5 min** | ❌ à faire |

## Mise en route

- `torch` n'est pas dans le Python système de la machine d'origine : créer un venv
  (`python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`).
- Tests : `pytest tests -q` → **213 tests, ~8 s**. Tous doivent rester verts.
- Lancer Python avec `PYTHONDONTWRITEBYTECODE=1` si l'on ne veut pas de `__pycache__`.
- `checkpoints/`, `*.pt`, `*.png`, `*.pdf` sont **gitignorés** : un clone neuf n'a **aucun
  checkpoint**. Il faut `python main_step3.py` (~40 min sur CPU) avant `main_step4.py`
  et `main_step5.py`, ou `main_step5.py --train-if-missing` (réseau rapide, peu précis).
- `python main_step3.py --test` = run court (~1-2 min) pour vérifier qu'il n'y a pas de bug.

## Ce qu'il reste à faire (par priorité)

### 1. Présentation de 5 minutes (livrable obligatoire, rien n'existe)
Aucun support n'est dans le dépôt. Matière disponible :
- adimensionnement (tableau du README) et pourquoi α disparaît ;
- `create_graph=True`, activation C², piège de la solution triviale (Étape 2) ;
- courbes d'entraînement (`step3_training_losses.png`) et comparaison de pondérations ;
- validation : `step4_validation.png`, tableau MSE / L2 par instant du README ;
- démonstrateur : cartes PINN / référence / erreur, avertissement « mauvais objet ».
Les limites à assumer sont listées dans la section Roadmap du README.

### 2. Qualité du PINN (optionnel mais c'est là que sont les gains)
Résultat nominal : L2 globale **12,9 %**, **10,8 % de points avec T\*<0**, erreur forte à
t\*=0 (créneau, Gibbs) et relative croissante aux temps longs (champ faible).
Pistes **non traitées** :
- sortie bornée ou contrainte dure sur l'IC (supprimer T\*<0 et Gibbs) ;
- Residual Adaptive Resampling (cité par le sujet) ;
- tester `lr_annealing` et des poids 50/50/1 (jamais évalués : runs interrompus) ;
- suréchantillonner l'IC près du bord de l'objet (l'objet ne couvre que ~7 % du carré).

### 3. `grad_norm` est cassé (bug connu, non corrigé)
Mesuré : erreur L2 **99 %**, effondrement sur T\*≡0 ; dans l'historique `w_ic` passe de 1
à 0,001 pendant que la perte IC reste à 0,065. La cause (boucle de rétroaction
w ∝ 1/‖∇L‖ ?) n'est **qu'une hypothèse**. Soit corriger (ex. plancher sur `w_ic`),
soit le laisser documenté comme inutilisable. Ne pas en faire le défaut.

### 4. Démonstrateur : limites assumées
- Un réseau par objet : changer la géométrie impose un ré-entraînement.
  Alternative lourde : PINN paramétré par (forme, centre, rayon) en entrée.
- Préréglage « Standard » : **~13 min** (759 s mesurées), 55 % d'erreur sur un carré
  décentré ; « Rapide » : 44 s, 74 %. Ne pas annoncer des durées non mesurées.
- Aucun checkpoint pré-entraîné n'est versionné. Décider si l'on en commite un
  (≈5 Mo) pour que la démo marche après un simple `git clone`.

## Pièges et conventions

- **Style** : commentaires et docstrings en **français**, pédagogiques (le *pourquoi*
  physique/mathématique, shapes `(N, 1)`), comme dans le code existant.
- **Ne pas lancer plusieurs entraînements en parallèle** : 5 runs simultanés ont pris
  ~40 min chacun au lieu de ~10 (surcharge CPU). Un run à la fois.
- **Ne pas changer les défauts de `LossConfig` (1/1/1)** : des tests s'appuient sur des
  poids neutres. Le défaut 10/10/1 est porté par la CLI de `main_step3.py` et par
  `DEMO_LOSS_CONFIG` dans `src/demo.py`.
- **Gradio** : `gr.State(model)` échoue (un `nn.Module` est pris pour un callable de
  chargement) → utiliser `gr.State(lambda: ...)`, comme dans `src/gradio_app.py`.
  `matplotlib.use("Agg")` est obligatoire (Gradio sert depuis des threads).
- **Convention de grille** : tous les champs sont indexés `[i_x, j_y]` (`indexing="ij"`) ;
  les `imshow` utilisent `.T`. Ne pas la mélanger.
- **Solveur DF** : explicite, stable pour r = Δt*/h² ≤ 0,2. Ne pas augmenter ce ratio
  au-delà de 0,25 (explosion). Le solveur est validé (ordre 2) : ne pas le « corriger »
  pour faire coller le PINN.
- **Pas de métrique non mesurée** dans le README : chaque chiffre y vient d'un run réel.
  Si un chiffre change (nouveau run nominal), mettre à jour README **et** ce fichier.
- Le README précise quels résultats ont été obtenus avec quels budgets ; les comparer
  à budget égal (3 000 Adam + 100 L-BFGS pour les comparaisons de pondération,
  5 000 + 500 pour le run nominal).

## Cartographie rapide du code

`config.py` (DataConfig, ModelConfig, LossConfig, TrainConfig) · `src/sampling.py`
(IC/BC/résidu) · `src/models.py` (PINN, `architecture()` / `from_architecture()`) ·
`src/physics.py` (autograd, résidu) · `src/losses.py` · `src/trainer.py` (Adam →
L-BFGS, checkpoints avec architecture + `data_cfg`) · `src/fd_solver.py` (référence) ·
`src/inference.py` (recharger un checkpoint, même ancien) · `src/validation.py`
(MSE, L2, figure) · `src/demo.py` (logique de la démo, testable sans navigateur) ·
`src/gradio_app.py` (câblage des widgets) · `main_step1..5.py` (points d'entrée).
