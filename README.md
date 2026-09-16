# PINN Thermal 2D — Diffusion thermique instationnaire

Modélisation de la **diffusion thermique 2D instationnaire** dans une pièce
(1 m × 1 m) par **Physics-Informed Neural Networks (PINNs)**.

Équation adimensionnée cible :

$$
\frac{\partial T^*}{\partial t^*}
=
\frac{\partial^2 T^*}{\partial x^{*2}}
+
\frac{\partial^2 T^*}{\partial y^{*2}}
$$

avec $(x^*, y^*) \in [0,1]^2$, $t^* \in [0,\ t^*_{\max}]$, $T^* \in [0,1]$.

> **Pédagogie / soutenance** — chaque module (`config.py`, `src/utils.py`,
> `src/sampling.py`, `src/models.py`, `src/physics.py`, `src/losses.py`,
> `src/trainer.py`, `main_step1.py`, `main_step2.py`, `main_step3.py`)
> documente en français le *pourquoi*
> physique ou mathématique de chaque bloc (adimensionnement Fourier,
> `create_graph=True` pour les dérivées secondes, activation $C^2$
> obligatoire, piège de la solution triviale, closure L-BFGS, etc.),
> pas seulement
> le *comment*. Les shapes des tenseurs sont systématiquement indiquées
> sous la forme `(N, 1) = (batch_size, n_features)`.

---

## Structure du dépôt

```
PINNs/
├── .gitignore
├── README.md
├── requirements.txt
├── pytest.ini
├── config.py              # DataConfig (physique) + ModelConfig + LossConfig + TrainConfig
├── main_step1.py          # Étape 1 — génération, validation, export figure
├── main_step2.py          # Étape 2 — réseau, autograd, loss, export figure
├── main_step3.py          # Étape 3 — entraînement hybride Adam → L-BFGS
├── src/
│   ├── __init__.py
│   ├── utils.py           # Sobol / LHS, masques, get_device, checkpoints, plots
│   ├── sampling.py        # sample_ic, sample_bc, sample_residual
│   ├── models.py          # PINN : MLP à activations C², normalisation [-1,1]
│   ├── physics.py         # autograd, résidu PDE, solution analytique
│   ├── losses.py          # L = w_ic·L_ic + w_bc·L_bc + w_res·L_res
│   └── trainer.py         # PINNTrainer : Adam + scheduler → L-BFGS, suivi, checkpoints
└── tests/
    ├── test_sampling.py   # Invariants Étape 1
    ├── test_models.py     # Architecture (Étape 2)
    ├── test_physics.py    # Autograd & résidu (Étape 2)
    ├── test_losses.py     # Loss multi-objectif (Étape 2)
    └── test_trainer.py    # Boucle hybride, early stopping, checkpoints (Étape 3)
```

---

## Adimensionnement (à retenir pour la soutenance)

| Grandeur | Formule | Notes |
|----------|--------|-------|
| $x^*$ | $x / L_x$ | $L_x = 1\,\mathrm{m}$ → $x^* \in [0,1]$ |
| $y^*$ | $y / L_y$ | $L_y = 1\,\mathrm{m}$ → $y^* \in [0,1]$ |
| $t^*$ | $\alpha t / L_{\mathrm{ref}}^2$ | **nombre de Fourier** ; $t^* \in [0,\ t^*_{\max}]$, $t^*_{\max} \approx 0.1$ |
| $T^*$ | $(T - T_{\mathrm{amb}}) / \Delta T$ | $\Delta T = T_{\mathrm{obj}} - T_{\mathrm{amb}} = 60^\circ\mathrm{C}$ |

La diffusivité $\alpha = 2\times 10^{-5}\,\mathrm{m}^2/\mathrm{s}$ (ordre de
grandeur de l'air) fixe le temps de référence :

$$
t_{\mathrm{ref}} = \frac{L^2}{\alpha} = 50\,000\,\mathrm{s}
\qquad\Rightarrow\qquad
t^*_{\max} = \frac{t_{\max}}{t_{\mathrm{ref}}} = \frac{5000}{50000} = 0.1
$$

**Piège classique** : échantillonner $t^*$ dans $[0, 1]$ au lieu de
$[0,\ t^*_{\max}]$. Cela reviendrait à simuler un horizon 10× trop long.

### Conditions

| Type | Contrainte adimensionnée |
|------|--------------------------|
| **IC** ($t^*=0$) | $T^*=1$ dans l'objet chaud (disque $r^*=0.15$ centré), $T^*=0$ à l'extérieur |
| **BC** (4 parois) | $T^*=0$ (Dirichlet à $T_{\mathrm{amb}}$), $t^* \in [0,\ t^*_{\max}]$ |
| **Résidu PDE** | $(x^*,y^*)\in(0,1)^2$, $t^*\in(0,t^*_{\max})$, $r=\partial_{t^*}T^*-\Delta^*T^*\approx 0$ |

### Pourquoi `requires_grad=True` sur $(x^*, y^*, t^*)$ ?

Le résidu PDE est construit par **autograd** PyTorch (voir
[Étape 2](#étape-2--architecture-du-pinn--dérivation-automatique)) :

```text
T_pred = network(x*, y*, t*)            # shape (N, 1)
∂T*/∂t*   = grad(T_pred, t*)            # dérivée 1ʳᵉ
∂T*/∂x*   = grad(T_pred, x*)
∂²T*/∂x*² = grad(∂T*/∂x*, x*)           # dérivée 2ⁿᵈᵉ (laplacien)
residual  = ∂T*/∂t* − (∂²T*/∂x*² + ∂²T*/∂y*²)
```

Sans `requires_grad=True` sur les entrées, `torch.autograd.grad` lève une
erreur. À l'inverse, la **cible** $T^*$ (IC/BC) a `requires_grad=False` :
c'est une constante physique, pas une variable d'optimisation.

---

## Installation

```bash
git clone <url-du-repo> PINNs
cd PINNs

python -m venv venv
source venv/bin/activate          # Windows : venv\Scripts\activate

pip install -r requirements.txt
```

**Dépendances** : `torch`, `numpy`, `matplotlib`, `scipy`, `gradio`, `pytest`,
`tqdm` (+ `tensorboard` optionnel pour la visualisation des courbes).

---

## Étape 1 — Adimensionnement & Échantillonnage

```bash
python main_step1.py
```

Options utiles :

```bash
python main_step1.py --method sobol --seed 42
python main_step1.py --method lhs --n-ic 500 --n-bc 500 --n-res 2000   # run rapide
python main_step1.py --out collocation_points.png
```

### Budgets par défaut (`DataConfig`)

| Ensemble | Symbole | N | Contrainte | Shape tenseurs |
|----------|---------|---|------------|----------------|
| Condition initiale | `N_ic` | 2 000 | $t^*=0$, $T^*\in\{0,1\}$ | `(2000, 1)` |
| Conditions aux limites | `N_bc` | 2 000 | $T^*=0$, 4 parois, $t^*\le t^*_{\max}$ | `(2000, 1)` |
| Résidu PDE | `N_res` | 20 000 | intérieur, $t^*\in(0,t^*_{\max})$ | `(20000, 1)` |

### Logs attendus (invariants)

```
[OK] IC  : N=2000, t*=0, T*∈[0.00, 1.00]  (hot=…, cold=…)
[OK] BC  : N=2000, T*=0, t*∈[0.0001, 0.0999] (t*_max=0.1000), walls L/R/B/T = 500/500/500/500
[OK] RES : N=20000, (x*,y*)∈(0,1)², t*∈[0.0000, 0.1000] (t*_max=0.1000)
```

### Tests

```bash
pytest tests/ -v
```

Les tests verrouillent notamment :
- $t^* = 0$ pour l'IC, $T^* \in \{0,1\}$ (objet chaud)
- $T^* = 0$ pour les BC
- $t^* \in [0,\ t^*_{\max}\approx 0.1]$ (pas $[0,1]$) pour BC et résidu
- shapes `(N, 1)` et `requires_grad=True` sur les coordonnées

---

## Étape 2 — Architecture du PINN & dérivation automatique

```bash
python main_step2.py
```

Options utiles :

```bash
python main_step2.py --layers 6 --neurons 128 --activation sin
python main_step2.py --w-ic 10 --w-bc 10 --w-res 1
python main_step2.py --n-ic 500 --n-bc 500 --n-res 2000   # run rapide
```

> Cette étape **n'entraîne pas** le réseau : elle construit et **valide**
> la machinerie que l'Étape 3 optimisera.

### Architecture $T_\theta(x^*, y^*, t^*)$

| Élément | Choix | Pourquoi |
|---------|-------|----------|
| Topologie | `3 → 64 → 64 → 64 → 64 → 1` (12 801 param.) | MLP dense : la solution de la chaleur est lisse, pas besoin de convolutions (il n'y a pas de grille) |
| Activation | `tanh` (aussi `sin`, `gelu`, `softplus`) | **doit être $C^2$** : le résidu contient $\partial^2 T^*/\partial x^{*2}$ |
| Normalisation | affine fixe vers $[-1, 1]$ | $x^*,y^* \in [0,1]$ mais $t^* \in [0,\ 0.1]$ : sans recentrage, le temps est 10× moins « visible » |
| Sortie | linéaire (pas de sigmoïde) | une saturation bloquerait les gradients pile sur $T^*=0$ et $T^*=1$ |
| Interdits | ni BatchNorm ni Dropout | $T_\theta$ doit rester une fonction déterministe du seul point $(x^*,y^*,t^*)$ |

**Pourquoi ReLU est bannie** — ReLU est affine par morceaux, donc sa dérivée
seconde est nulle presque partout. Le laplacien vaudrait exactement $0$ et le
résidu se réduirait à $\partial T^*/\partial t^*$ : le réseau apprendrait un
champ **gelé**. `build_activation("relu")` lève donc une `ValueError`, et
`main_step2.py` le démontre numériquement :

```
[info] |Δ*T*| moyen — tanh : 2.683e-01   ReLU : 0.000e+00
```

### Dérivation automatique (`src/physics.py`)

```python
T     = model(x_star, y_star, t_star)      # (N, 1)
T_t   = grad(T,   t_star)                  # ∂T*/∂t*
T_x   = grad(T,   x_star, create_graph=True)
T_xx  = grad(T_x, x_star)                  # ∂²T*/∂x*²   ← 2ᵉ dérivée
residual = T_t - (T_xx + T_yy)
```

Deux points à savoir expliquer en soutenance :

- **`create_graph=True`** sur la dérivée première est indispensable : sans
  lui, `T_x` est détaché du graphe et la dérivée seconde lève une erreur.
  Il rend aussi la loss résidu différentiable par rapport à $\theta$.
- **`grad_outputs=ones`** : `autograd.grad` calcule un produit
  vecteur-jacobienne. Comme chaque point de collocation est traité
  indépendamment, les termes croisés sont nuls et l'on récupère
  exactement le vecteur des dérivées ponctuelles — tout le batch en un appel.

**Aucun coefficient $\alpha$ n'apparaît** dans le résidu : c'est le bénéfice
de l'adimensionnement de l'Étape 1, $\alpha$ étant absorbé dans $t^*$.

### Validation de l'autograd (le cœur de l'étape)

Un bug d'autograd ne plante pas : il produit silencieusement un résidu faux,
et le réseau converge très bien… vers une autre équation. D'où trois contrôles :

| Contrôle | Principe | Résultat attendu |
|----------|----------|------------------|
| Solution analytique | $T^*=\sin(\pi x^*)\sin(\pi y^*)e^{-2\pi^2 t^*}$ vérifie l'EDP exactement | $r_{\text{rel}} \approx 10^{-16}$ (float64) |
| Contre-épreuve | $T = x^{*2}+y^{*2}+t^*$ donne $r = 1 - 4 = -3$ | $r = -3$ exactement |
| Différences finies | compare l'autograd du **vrai réseau** à $[T(x+h)-T(x-h)]/2h$ | écart $\lesssim 10^{-8}$ |

La contre-épreuve est indispensable : sans elle, un `pde_residual` qui
renverrait bêtement zéro passerait le premier test avec les honneurs.

### Loss multi-objectif

$$
\mathcal{L}(\theta) = w_{IC}\,\mathcal{L}_{IC} + w_{BC}\,\mathcal{L}_{BC} + w_{res}\,\mathcal{L}_{res}
$$

| Terme | Définition | Rôle |
|-------|-----------|------|
| $\mathcal{L}_{IC}$ | $\mathrm{MSE}\big(T_\theta(x^*,y^*,0),\ T^*_{\text{cible}}\big)$ | supervisé — le terme le plus **raide** (cible en créneau) |
| $\mathcal{L}_{BC}$ | $\mathrm{MSE}\big(T_\theta(\partial\Omega,t^*),\ 0\big)$ | supervisé — murs à $T_{\text{amb}}$ |
| $\mathcal{L}_{res}$ | $\mathrm{MSE}\big(\partial_{t^*}T^* - \Delta^* T^*,\ 0\big)$ | **aucune donnée cible** : la vérité est l'équation |

**Pourquoi les trois termes sont indispensables ensemble** — $T^* \equiv 0$
vérifie *parfaitement* l'EDP et les BC. C'est la **solution triviale**, le
piège classique du PINN : seul $\mathcal{L}_{IC}$ l'écarte. Les tests le
démontrent numériquement (`test_zero_field_satisfies_the_pde`).

Corollaire vérifié dans `test_physics.py` : le **biais de la couche de sortie
ne reçoit aucun gradient du résidu seul**, car l'opérateur de la chaleur
annule les constantes. L'EDP ne détermine la solution qu'à une constante
additive près — ce sont l'IC et les BC qui la fixent.

### Logs attendus

```
[OK] forward : sortie (16, 1), dtype=torch.float32, grad_fn=AddmmBackward0
[OK] analytique : résidu relatif = 1.157e-16 (float64, N=4096)
[OK] contre-épreuve : T=x²+y²+t donne bien r = −3
[OK] diff. finies : ∂T*/∂x* à 2.80e-09, ∂²T*/∂x*² à 2.14e-07
[OK] loss  : L=1.2527e+01  |  L_ic=1.3374e-01  L_bc=3.3402e-02  L_res=1.2360e+01
[OK] backward : 10/10 tenseurs de paramètres reçoivent un gradient
```

**Lecture du déséquilibre initial** : $\mathcal{L}_{res}$ domine de deux ordres
de grandeur. $\mathcal{L}_{IC}$ et $\mathcal{L}_{BC}$ comparent des *valeurs*,
tandis que $\mathcal{L}_{res}$ compare des *dérivées secondes* — dériver deux
fois amplifie par le carré des fréquences du réseau. Avec des poids neutres,
l'optimiseur servirait donc d'abord le résidu… qui admet la solution triviale.
C'est exactement ce qui motive la **pondération dynamique de l'Étape 3**.

---

## Étape 3 — Entraînement hybride Adam → L-BFGS

```bash
python main_step3.py            # run nominal : 5 000 epochs Adam → 500 itérations L-BFGS
python main_step3.py --test     # run court de validation : 500 + 50, budgets réduits
```

> **Commencer par `--test`** : ce mode réduit les budgets
> (500 epochs Adam + 50 itérations L-BFGS, échantillons 500/500/2000)
> et ajoute un **bilan mémoire** (RSS avant/après, VRAM restituée,
> historique sans tenseur PyTorch) — de quoi prouver l'absence de bugs
> et de fuite avant d'engager un run long.

### Pourquoi deux phases ?

| Phase | Optimiseur | Rôle | Limite |
|-------|-----------|------|--------|
| 1 | `Adam` + scheduler (`ReduceLROnPlateau` ou `CosineAnnealingLR`) | Exploration robuste : tolère les gradients raides des dérivées secondes, s'échappe des mauvais bassins | Oscille autour du minimum (plafond ~1e-4/1e-5), ne s'y installe jamais |
| 2 | `L-BFGS` (`line_search_fn='strong_wolfe'`) | Affinage quasi-Newton : mémoire des courbures passées → convergence quasi exacte | Trompeur loin du minimum — d'où l'ordre **Adam puis L-BFGS** |

Deux règles non négociables, implémentées dans `src/trainer.py` :

- **La closure L-BFGS** : `optimizer.step(closure)` exige une fonction qui
  remet les gradients à zéro, recalcule la loss (la line search évalue
  plusieurs points d'essai), lance `backward()` et **renvoie** la loss.
- **Full-batch obligatoire** : L-BFGS mémorise une courbure *entre* deux
  évaluations — tout mini-batch aléatoire casserait sa mémoire et sa
  line search. Le run est de plus déterministe à seed fixée.

### Pondération des pertes (configurable)

| Schéma | Formule | Notes |
|--------|---------|-------|
| `fixed` (défaut) | $w_{ic}, w_{bc}, w_{res}$ de `LossConfig` | Référence reproductible |
| `grad_norm` | $w_i \propto 1/\lVert\nabla_\theta \mathcal{L}_i\rVert_2$, normalisé (max = 1) | Équilibre les tirages ; le résidu dominant est dégonflé (esprit GradNorm) |
| `lr_annealing` | $w_i \propto \hat{w}_i^{-1}$ avec $\hat{w}_i = \max_\theta\lvert\nabla_\theta \mathcal{L}_i\rvert / \mathrm{mean}_\theta\lvert\nabla_\theta \mathcal{L}_i\rvert$ sur la 1ʳᵉ couche | Wang, Teng & Perdikaris (ICML 2021) |

Les facteurs adaptatifs se multiplient aux poids statiques
($w_{eff} = w_{config} \times w_{adaptatif}$), se mettent à jour toutes
les `weighting_update_every` epochs (après un `warmup` optionnel) et sont
**gelés pendant L-BFGS** : un quasi-Newton exige un objectif stationnaire.

```bash
python main_step3.py --weighting grad_norm
python main_step3.py --weighting lr_annealing --weighting-every 50 --weighting-warmup 200
python main_step3.py --w-ic 10 --w-bc 10 --w-res 1        # poids fixes
```

### Suivi, checkpoints, early stopping

| Artefact | Contenu |
|----------|---------|
| `checkpoints/best_model.pt` | Poids + optimiseurs au **record** de la métrique surveillée (`total` ou `residual`) |
| `checkpoints/last_model.pt` | Idem + scheduler + historique complet — écrit périodiquement (`checkpoint_every`) et en fin de run |
| `checkpoints/history.json` | Historique complet (JSON) pour post-mortem |
| `step3_training_losses.png` | Figure 2×2 : perte totale (log), termes bruts, poids effectifs, lr |

L'historique enregistre à chaque itération : $\mathcal{L}_{ic}$,
$\mathcal{L}_{bc}$, $\mathcal{L}_{res}$, $\mathcal{L}_{total}$, les poids
effectifs, le lr et le temps écoulé — **uniquement en floats Python**
(stocker des tenseurs y retiendrait les graphes autograd → fuite mémoire
garantie). Progression **tqdm** par défaut, **TensorBoard** en option
(`--tensorboard`, puis `tensorboard --logdir runs/`).

L'**early stopping** (patience, min_delta, métrique au choix) coupe le run
si la métrique n'améliore plus ; un garde-fou restaure le meilleur
checkpoint et arrête proprement en cas de NaN ou d'échec de line search.

### Logs attendus (`--test`)

```
[adam     1/500] L=1.253e+01 | ic=1.36e-01 bc=3.32e-02 res=1.24e+01 | w=(1.00,1.00,1.00) | lr=1.00e-03 | t+0.1 s
[adam   500/500] L=6.655e-02 | ic=6.53e-02 bc=1.01e-03 res=2.48e-04 | w=(1.00,1.00,1.00) | lr=1.00e-03 | t+24.8 s
[lbfgs    1/50] L=6.462e-02 | ic=6.22e-02 bc=3.98e-04 res=2.05e-03 | evals=23 | t+26.0 s
[lbfgs   50/50] L=3.949e-02 | ic=3.64e-02 bc=6.91e-04 res=2.44e-03 | evals=1134 | t+1 min 21 s
```

Lecture : Adam écrase d'abord le résidu (12.4 → 2.5e-4) ; il reste alors le
terme raide $\mathcal{L}_{ic}$ (le créneau de l'objet chaud), que L-BFGS
raffine (6.7e-2 → 3.9e-2). Le compteur `evals` rappelle le vrai coût de
L-BFGS : chaque itération évalue la closure plusieurs fois (line search).

### Tests

```bash
pytest tests/ -v        # 143 tests, dont 25 dédiés à l'Étape 3
```

`tests/test_trainer.py` verrouille : exécution des deux phases, historique
sans tenseur (anti-fuite), `best_model.pt` = minimum de la métrique,
déclenchement de l'early stopping, bornes et normalisation des poids
adaptatifs, scheduler, roundtrip de checkpoint, figure.

---

## Roadmap

| Étape | Contenu | Statut |
|-------|---------|--------|
| **1** | Adimensionnement & échantillonnage IC/BC/résidu | ✅ |
| **2** | Architecture du réseau & loss PINN (autograd) | ✅ |
| **3** | Entraînement hybride Adam → L-BFGS, checkpoints | ✅ |
| **4** | Démonstrateur interactif Gradio | 🔜 |
| **5** | Présentation (5 min) | 🔜 |

---

## Configuration rapide

```python
from config import DataConfig, ModelConfig, LossConfig, TrainConfig
from src.sampling import sample_ic, sample_bc, sample_residual
from src.models import PINN
from src.physics import pde_residual
from src.losses import pinn_loss
from src.trainer import PINNTrainer
from src.utils import get_device

cfg = DataConfig()          # Lx=Ly=1, alpha=2e-5, N_ic=2000, t*_max=0.1, …
print(cfg.summary())

# --- Étape 1 : points de collocation ---
ic  = sample_ic(cfg)        # dict: x_star, y_star, t_star, T_star   shapes (N_ic, 1)
bc  = sample_bc(cfg)        # + wall                                 shapes (N_bc, 1)
res = sample_residual(cfg)  # x_star, y_star, t_star                 shapes (N_res, 1)

# --- Étape 2 : réseau, résidu, loss ---
model = PINN.from_config(cfg, ModelConfig())   # bornes de normalisation ← cfg
r = pde_residual(model, res["x_star"], res["y_star"], res["t_star"])  # (N_res, 1)

terms = pinn_loss(model, ic, bc, res, LossConfig(w_ic=1.0, w_bc=1.0, w_res=1.0))
print(terms)                # L=…  |  L_ic=…  L_bc=…  L_res=…
terms.total.backward()      # ∂L/∂θ prêt pour l'optimiseur de l'Étape 3

# --- Étape 3 : entraînement hybride Adam → L-BFGS ---
trainer = PINNTrainer(
    model, ic, bc, res,
    LossConfig(w_ic=1.0, w_bc=1.0, w_res=1.0),
    TrainConfig(adam_epochs=5000, lbfgs_iterations=500, weighting="fixed"),
    device=get_device("auto"),          # cuda > mps > cpu
    checkpoint_dir="checkpoints",
)
result = trainer.fit()                  # FitResult : record, checkpoints, arrêt…
print(result.summary())
trainer.restore_best_weights()          # revenir au best_model.pt
```

---

## Licence

Projet académique / démonstrateur PINN — usage libre pour l'enseignement et la R&D.
