# PINN Thermal 2D — Diffusion thermique instationnaire

Modélisation de la **diffusion thermique 2D instationnaire** dans une pièce (1 m × 1 m) par **Physics-Informed Neural Networks (PINNs)**.

Équation adimensionnée cible :

$$
\frac{\partial T^*}{\partial t^*} = \frac{\partial^2 T^*}{\partial x^{*2}} + \frac{\partial^2 T^*}{\partial y^{*2}}
$$

avec $(x^*, y^*) \in [0,1]^2$, $T^* \in [0,1]$.

---

## Structure du dépôt

```
PINNs/
├── .gitignore
├── README.md
├── requirements.txt
├── config.py              # DataConfig : physique, géométrie, budgets d'échantillonnage
├── main_step1.py          # Étape 1 — génération, validation, export figure
├── src/
│   ├── __init__.py
│   ├── utils.py           # Sobol / LHS, masques géométriques
│   └── sampling.py        # sample_ic, sample_bc, sample_residual
└── tests/
    └── test_sampling.py   # Tests unitaires PyTest
```

---

## Adimensionnement

| Grandeur | Formule | Notes |
|----------|--------|-------|
| $x^*$ | $x / L_x$ | $L_x = 1\,\mathrm{m}$ |
| $y^*$ | $y / L_y$ | $L_y = 1\,\mathrm{m}$ |
| $t^*$ | $t / t_{\max}$ | normalisé sur $[0,1]$ pour le réseau |
| $T^*$ | $(T - T_{\mathrm{amb}}) / (T_{\mathrm{obj}} - T_{\mathrm{amb}})$ | $T_{\mathrm{amb}}=20^\circ\mathrm{C}$, $T_{\mathrm{obj}}=80^\circ\mathrm{C}$ |

La diffusivité $\alpha = 2\times 10^{-5}\,\mathrm{m}^2/\mathrm{s}$ (ordre de grandeur de l'air) définit le temps de référence $t_{\mathrm{ref}} = L^2/\alpha$ et le nombre de Fourier $Fo = \alpha t / L^2$.

Conditions aux limites Dirichlet adimensionnées : **$T^* = 0$** sur les 4 parois (température ambiante).  
Condition initiale : **$t^* = 0$**, $T^* = 0$ (pièce initialement à $T_{\mathrm{amb}}$).

---

## Installation

```bash
# Clone
git clone <url-du-repo> PINNs
cd PINNs

# Environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Dépendances
pip install -r requirements.txt
```

**Dépendances principales** : `torch`, `numpy`, `matplotlib`, `scipy`, `gradio`, `pytest`.

---

## Étape 1 — Adimensionnement & Échantillonnage

Génère les points de collocation, valide les invariants et exporte `collocation_points.png` :

```bash
python main_step1.py
```

Options utiles :

```bash
python main_step1.py --method sobol --seed 42
python main_step1.py --method lhs --n-ic 2000 --n-bc 2000 --n-res 20000
python main_step1.py --out collocation_points.png
```

### Budgets par défaut (`DataConfig`)

| Ensemble | Symbole | N | Contrainte |
|----------|---------|---|------------|
| Condition initiale | `N_ic` | 2 000 | $t^* = 0$ |
| Conditions aux limites | `N_bc` | 2 000 | $T^* = 0$, 4 parois |
| Résidu PDE | `N_res` | 20 000 | intérieur $(0,1)^3$ |

Les coordonnées $(x^*, y^*, t^*)$ sont des tenseurs PyTorch avec `requires_grad=True` pour l'autograd des dérivées PDE.

### Tests

```bash
pytest tests/ -v
```

Les tests vérifient notamment :
- `t* = 0` pour l'IC
- `T* = 0` pour les BC
- shapes `(N, 1)` conformes aux budgets
- `requires_grad=True` sur les coordonnées
- points BC strictement sur le bord, résidu strictement intérieur

---

## Roadmap

| Étape | Contenu | Statut |
|-------|---------|--------|
| **1** | Adimensionnement & échantillonnage IC/BC/résidu | ✅ |
| **2** | Architecture du réseau & loss PINN | 🔜 |
| **3** | Entraînement & checkpoints | 🔜 |
| **4** | Démonstrateur interactif Gradio | 🔜 |
| **5** | Présentation (5 min) | 🔜 |

---

## Configuration rapide

```python
from config import DataConfig
from src.sampling import sample_ic, sample_bc, sample_residual

cfg = DataConfig()          # Lx=Ly=1, alpha=2e-5, N_ic=2000, …
print(cfg.summary())

ic  = sample_ic(cfg)        # dict: x_star, y_star, t_star, T_star
bc  = sample_bc(cfg)        # + wall
res = sample_residual(cfg)  # x_star, y_star, t_star
```

---

## Licence

Projet académique / démonstrateur PINN — usage libre pour l'enseignement et la R&D.
