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
> `src/sampling.py`, `main_step1.py`) documente en français le *pourquoi*
> physique ou mathématique de chaque bloc (adimensionnement Fourier,
> `requires_grad` pour l'autograd, masque objet chaud, etc.), pas seulement
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
├── config.py              # DataConfig : physique, géométrie, adimensionnement
├── main_step1.py          # Étape 1 — génération, validation, export figure
├── src/
│   ├── __init__.py
│   ├── utils.py           # Sobol / LHS, masques géométriques, to_tensor
│   └── sampling.py        # sample_ic, sample_bc, sample_residual
└── tests/
    └── test_sampling.py   # Tests unitaires PyTest (invariants Étape 1)
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

À l'Étape 2, le résidu PDE est construit par **autograd** PyTorch :

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

**Dépendances** : `torch`, `numpy`, `matplotlib`, `scipy`, `gradio`, `pytest`.

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

## Roadmap

| Étape | Contenu | Statut |
|-------|---------|--------|
| **1** | Adimensionnement & échantillonnage IC/BC/résidu | ✅ |
| **2** | Architecture du réseau & loss PINN (autograd) | 🔜 |
| **3** | Entraînement & checkpoints | 🔜 |
| **4** | Démonstrateur interactif Gradio | 🔜 |
| **5** | Présentation (5 min) | 🔜 |

---

## Configuration rapide

```python
from config import DataConfig
from src.sampling import sample_ic, sample_bc, sample_residual

cfg = DataConfig()          # Lx=Ly=1, alpha=2e-5, N_ic=2000, t*_max=0.1, …
print(cfg.summary())

ic  = sample_ic(cfg)        # dict: x_star, y_star, t_star, T_star   shapes (N_ic, 1)
bc  = sample_bc(cfg)        # + wall                                 shapes (N_bc, 1)
res = sample_residual(cfg)  # x_star, y_star, t_star                 shapes (N_res, 1)

# Prêt pour l'Étape 2 : res["x_star"].requires_grad == True
```

---

## Licence

Projet académique / démonstrateur PINN — usage libre pour l'enseignement et la R&D.
