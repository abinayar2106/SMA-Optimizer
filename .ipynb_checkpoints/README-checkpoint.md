# SMA-Optimizer for Ni-Ti-Hf alloys

**An interpretable, generalizable ML framework for NiTi-based SMAs: identifying compositional and processing parameters that govern kinematic compatibility (λ₂) and volumetric stability (ΔV), with composition-space mapping of the triple constraint — T > 200°C, |λ₂ − 1| < 0.01, ΔV ≈ 0
Future Goal: To apply this for Ni-Ti-Cu/Pt/Pd alloys
---

## Motivation

NiTiHf shape memory alloys (SMAs) are candidates for high-temperature solid-state actuation, but two conditions must be satisfied simultaneously:

1. **Transformation temperature** must exceed 200 °C for the target application window.
2. **Middle eigenvalue λ₂** of the transformation stretch matrix must be close to 1 — the crystallographic condition for kinematic compatibility, low hysteresis, and long cyclic life.

No systematic map of where both conditions are met across the NiTiHf composition space currently exists. This project builds that map, and interprets what drives it.

---

## Architecture

The pipeline has three arms, deliberately separating learned behavior from physics:

```
Composition + Processing Features
          │
          ├──► Arm 1: ML Surrogate (XGBoost / Random Forest)
          │         └── Predicts: Ms, Af, hysteresis, recoverable strain
          │
          ├──► Arm 2: Physics Engine (NumPy linear algebra)
          │         └── Fits lattice parameters from XRD data
          │             Builds transformation stretch matrix U analytically
          │             Computes eigenvalues → extracts λ₂
          │
          └──► Arm 3: Interpretability (SHAP)
                    └── Ranks feature influence on each output
                        Global summary + local composition-level explanations

                    ↓
         Ternary Overlay: T > 200°C ∩ |λ₂ − 1| < 0.01
                    ↓
         2–3 Curated Ideal Compositions
```

**Why not a pure neural network?**
A black-box model would estimate cyclic stability statistically. This architecture forces the model to predict intermediate crystal lattice parameters, which are then used to compute λ₂ analytically — guaranteeing that recommended compositions are crystallographically valid, not statistical anomalies.

---

## Features

### Inputs
| Feature | Description |
|---|---|
| Ni, Ti, Hf (at%) | Ternary composition | 
| VEC / e/a ratio | Valence electron concentration |
| Elemental bonding fractions | d-electron and bonding character |
| Atomic size mismatch | Variance in atomic radii (driven by Hf) |
| Pettifor scale | Phenomenological chemical ordering parameter |
| B2 austenite lattice parameter | XRD-derived, *a* (Å) |
| B19′ martensite parameters | *a*, *b*, *c* (Å) + monoclinic angle β (°) |
| Processing conditions | Aging time, aging temperature, thermo-mechanical history |
| Measurement technique | DSC vs. TMA flag |

### Outputs
| Target | Description |
|---|---|
| Ms, Mf, As, Af | Martensite/Austenite start and finish temperatures |
| Thermal hysteresis | Af − Ms |
| Recoverable strain | Maximum theoretical macroscopic strain from stretch matrix |
| λ₂ | Middle eigenvalue of transformation stretch matrix U |
| \|λ₂ − 1\| | Kinematic compatibility condition; primary cyclic stability metric |
| SHAP values | Per-feature importance for each output |

---

## Pipeline

| Step | Tool | Input | Output |
|---|---|---|---|
| 1. Structural surrogate | XGBoost / Random Forest | Composition + elemental features | Predicted lattice parameters |
| 2. Thermodynamic surrogate | XGBoost / Random Forest | Composition + processing features | Ms, Af, hysteresis, recoverable strain |
| 3. Kinematic engine | NumPy (`linalg.eig`) | Predicted lattice parameters | Stretch matrix U → λ₂ |
| 4. Interpretability | SHAP | Trained surrogates | Feature importance rankings per output |
| 5. Optimization loop | Python scripting | Millions of synthetic compositions | Filtered candidates: Ms > 200°C, \|λ₂ − 1\| < 0.01 |

Validation uses **leave-one-study-out cross-validation** to account for inter-study variability in the NASA SMA database.

---

## Repo Structure

```
NiTiHf-SMA-Optimizer/
├── data/
│   ├── raw/                  # NASA SMA database + XRD data
│   └── processed/            # Cleaned, feature-engineered dataset
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_arm1_ttrans.ipynb  # Transformation temperature surrogate
│   ├── 03_arm2_lambda.ipynb  # Eigenvalue engine
│   └── 04_arm3_shap.ipynb    # Interpretability analysis
├── src/
│   ├── arm1_ttrans.py        # ML surrogate (training + inference)
│   ├── arm2_lambda.py        # Physics-based λ₂ computation
│   ├── arm3_shap.py          # SHAP analysis and plots
│   └── pipeline.py           # End-to-end integration
├── outputs/
│   ├── ternary_maps/         # T and λ₂ overlay plots
│   ├── shap_plots/           # Feature importance visualizations
│   └── candidates.csv        # Final curated compositions
├── requirements.txt
└── README.md
```

---

## Data Source

NASA Shape Memory Alloy database. Composition, processing, structural (XRD), and thermodynamic data for NiTiHf alloys across a range of aging conditions and measurement techniques.

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost
- **Interpretability** — SHAP
- **Visualization** — matplotlib, plotly (ternary maps)
- **Validation** — leave-one-study-out cross-validation

---

## Status

In active development — part of an ongoing research project under Prof. K. S. Suresh, IIT Roorkee (Materials Informatics).

---

## References

1. Xue, D. et al. *Accelerated search for materials with targeted properties by adaptive design.* Nature Communications (2016).
2. Sehitoglu, H. et al. *AI-enabled materials discovery for narrow-hysteresis shape memory alloys.* (2022).
3. Trehern, W. et al. *Data-driven shape memory alloy discovery using Artificial Intelligence Materials Selection (AIMS) framework* (2022).
