# Geometric Algebra Transformer Systems (GATS)

**GATS** (`gats.py`) implements a multivector-based **Geometric Algebra (GA)** representation coupled with **linear attention** for explanatory analysis of consumer credit cycles.  
It is designed to **explain** the dynamics behind credit charge-offs, not just predict them.

> Paper: "Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis" by Sudjianto & Setiawan (2025).

---

## Contents

- [Why GA + Attention?](#why-ga--attention)
- [Mathematical Overview](#mathematical-overview)
- [Installation](#installation)
- [Data Schema](#data-schema)
- [Quickstart](#quickstart)
- [Full API](#full-api)
  - [`complete_ccci_ga_workflow`](#complete_ccci_ga_workflow)
  - [`GALinearAttentionModel`](#galinearattentionmodel)
  - [`GeometricAlgebraEmbedding`](#geometricalgebraembedding)
  - [`LinearAttention`](#linearattention)
  - [Heads: `LinearHead` and `MLPHead`](#heads-linearhead-and-mlphead)
  - [Training: `train_model`](#train_model)
  - [Analysis utilities](#analysis-utilities)
- [Reproducing Paper Figures (1–7)](#reproducing-paper-figures-17)
- [Interpretability Outputs](#interpretability-outputs)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Citations](#citations)

---

## Why GA + Attention?

- **Geometric Algebra embedding** augments raw variables with **bivectors** that quantify **rotational (feedback) relationships** (e.g., unemployment ↔ credit).
- **Linear attention** provides **time-varying weights** over a rolling window, selecting historical precedents similar to the current state.
- **Prediction heads**: choose **Linear** for clarity and additive attributions, or **MLP** for improved fit when accuracy is preferred.

---

## Mathematical Overview

Let the 4-D standardized macro state at time *t* be **X**<sub>*t*</sub> = [*u*<sub>*t*</sub>, *s*<sub>*t*</sub>, *r*<sub>*t*</sub>, *v*<sub>*t*</sub>]  
(unemployment, saving rate, PCE growth, revolving credit growth).

### GA embedding

We form a multivector:

**M**<sub>*t*</sub> = α<sub>0</sub> + Σ<sub>*i*=1</sub><sup>4</sup> α<sub>*i*</sub> **e**<sub>*i*</sub> + Σ<sub>(*i*,*j*)∈Π</sub> γ<sub>*ij*</sub>(*x*<sub>*t*,*i*</sub> - *x*<sub>*t*,*j*</sub>)(**e**<sub>*i*</sub> ∧ **e**<sub>*j*</sub>)

with planes Π = {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}.

### Linear attention (feature map φ(*x*) = LeakyReLU(*x*) + 1)

**Q**<sub>*t*</sub> = φ(**W**<sub>*Q*</sub> **M**<sub>*t*</sub>),  **K**<sub>*t*</sub> = φ(**W**<sub>*K*</sub> **M**<sub>*t*</sub>),  **V**<sub>*t*</sub> = **W**<sub>*V*</sub> **M**<sub>*t*</sub>

over a lookback window W<sub>*t*</sub> = {*t*-*L*, ..., *t*-1}.

Denote:
- **S**<sub>*t*</sub> = Σ<sub>τ∈W<sub>*t*</sub></sub> **K**<sub>τ</sub> **V**<sub>τ</sub><sup>⊤</sup>
- **Z**<sub>*t*</sub> = Σ<sub>τ∈W<sub>*t*</sub></sub> **K**<sub>τ</sub>

Then the normalized context is:

**O**<sub>*t*</sub> = (**Q**<sub>*t*</sub><sup>⊤</sup> **S**<sub>*t*</sub>) / (**Q**<sub>*t*</sub><sup>⊤</sup> **Z**<sub>*t*</sub> + ε)

*w*<sub>τ|*t*</sub> = (**Q**<sub>*t*</sub><sup>⊤</sup> **K**<sub>τ</sub>) / (Σ<sub>κ∈W<sub>*t*</sub></sub> **Q**<sub>*t*</sub><sup>⊤</sup> **K**<sub>κ</sub>)

### Heads

- **Linear**: ŷ<sub>*t*</sub> = **w**<sub>out</sub><sup>⊤</sup> **O**<sub>*t*</sub> + *b*
- **MLP**: ŷ<sub>*t*</sub> = **W**<sub>2</sub> σ(**W**<sub>1</sub> **O**<sub>*t*</sub> + **b**<sub>1</sub>) + *b*<sub>2</sub> (ReLU/GELU)

> Economic note: the **dot product** **Q**<sub>*t*</sub><sup>⊤</sup> **K**<sub>τ</sub> is a scalar **geometric similarity** between the current multivector state and historical states (suitable for weighting).

---

## Installation

```bash
# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A minimal `requirements.txt`:

```
numpy
pandas
torch
scikit-learn
matplotlib
seaborn
```

---

## Data Schema

CSV with columns (quarterly):

| column        | type   | description                                                |
|---------------|--------|------------------------------------------------------------|
| `quarter`     | str    | e.g. `1980Q1`                                              |
| `UNRATE`      | float  | unemployment rate (%)                                      |
| `PSAVERT`     | float  | personal saving rate (%)                                   |
| `PCE_QoQ`     | float  | QoQ % growth of PCE                                        |
| `REVOLSL_QoQ` | float  | QoQ % growth of revolving credit                           |
| `CORCACBS`    | float  | charge-off rate on consumer loans (%) **(target)**         |

---

## Quickstart

```python
# main.py
from gats import complete_ccci_ga_workflow

feature_cols = ["UNRATE", "PSAVERT", "PCE_QoQ", "REVOLSL_QoQ"]

results = complete_ccci_ga_workflow(
    csv_file="ccci_data.csv",
    feature_columns=feature_cols,
    target_column="CORCACBS",
    lookback=8,           # quarters of history attended
    train_frac=0.90,      # fraction used for train split
    epochs=300,           # training epochs
    head_type="mlp",      # "linear" for clarity, "mlp" for accuracy
    output_dir="./ccci_ga_analysis/"
)

print("Done. Key outputs:")
print("  Report:", results["report_path"])
print("  Figures:", results["comprehensive_results"]["file_paths"])
print("  Train RMSE:", results["performance"]["train_rmse"])
print("  Test  RMSE:", results["performance"]["test_rmse"])
print("  Overall R^2:", results["performance"]["comprehensive_r2"])
```

Run:

```bash
python main.py
```

Outputs (under `output_dir`):

- `results_analysis/figure1..figure7.png` (paper figures)
- `heatmaps/comprehensive_heatmaps.png`
- `comprehensive_ccci_report.md` (executive summary + findings)

---

## Full API

### `complete_ccci_ga_workflow`

End‑to‑end pipeline: load, scale, build model, train, evaluate, generate figures/report.

**Signature**
```python
complete_ccci_ga_workflow(
    csv_file: str = 'ccci_data.csv',
    feature_columns: list[str] | None = None,
    target_column: str = 'CORCACBS',
    lookback: int = 8,
    train_frac: float = 0.98,
    epochs: int = 100,
    specific_queries: list[int] | None = None,
    head_type: str = "linear",         # "linear" or "mlp"
    output_dir: str = "./ccci_ga_analysis/"
) -> dict
```

**Key parameters**

- `feature_columns`: order matters (maps to GA basis **e**<sub>1</sub>..**e**<sub>4</sub>).
- `target_column`: typically `CORCACBS`.
- `lookback`: attention window *L* (8–12 is typical).
- `train_frac`: temporal split (uses earliest `train_frac` fraction for training).
- `epochs`: training epochs for AdamW.
- `head_type`: `"linear"` (interpretable) vs `"mlp"` (higher accuracy).
- `specific_queries`: list of indices (post-lookback) for deep-dive attribution plots.
- `output_dir`: where all artifacts are saved.

**Returns** a dict with:
- `model` (trained `GALinearAttentionModel`), `scaler`, `predictions`, `attention`
- `performance` (RMSE/R²), `comprehensive_results` (fig paths), `heatmap_results`
- `query_analyses` (if requested), `insights`, `report_path`

---

### `GALinearAttentionModel`

```python
from gats import GALinearAttentionModel
model = GALinearAttentionModel(
    bivector_planes=None,      # default: all 6 unique pairs
    hidden_dim=32,             # attention state size
    lookback=8,
    eps=1e-6,
    include_query_side=True,   # enables query-only occlusion analysis
    head_type="linear",        # or "mlp"
    mlp_hidden=(64, 32),       # used if head_type="mlp"
    mlp_dropout=0.0,
    mlp_activation="relu",
    attention_entropy_threshold=1.5,
    attribution_threshold_factor=1.5
)
```

**Notes**
- `bivector_planes`: subset planes to increase sparsity/interpretability.
- `hidden_dim`: larger captures richer similarity but may overfit.
- `include_query_side`: enables query‑only component ablations in `interpretability`.

---

### `GeometricAlgebraEmbedding`

```python
from gats import GeometricAlgebraEmbedding
embed = GeometricAlgebraEmbedding(
    bivector_planes=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
)
```

- Produces multivectors of dimension `1 + 4 + len(planes)` = scalar + 4 vectors + bivectors.
- Trainable parameters: `scalar_bias`, `vector_scale`, `vector_bias`, `bivector_weight`.

---

### `LinearAttention`

- Projections: **W**<sub>*Q*</sub>, **W**<sub>*K*</sub>, **W**<sub>*V*</sub> (learnable `nn.Linear`).
- Feature map: φ(*x*) = LeakyReLU(*x*) + 1 keeps entries positive to stabilize normalization.
- Returns `(contexts, attention_weights)` where `attention_weights[:, t, :]` are the normalized *w*<sub>τ|*t*</sub>.

**Structured regularization (built into optimizer factory)**  
We apply stronger weight decay to **W**<sub>*Q*</sub>/**W**<sub>*K*</sub> than to **W**<sub>*V*</sub> to stabilize similarity while allowing informative values.

---

### Heads: `LinearHead` and `MLPHead`

- `LinearHead`: single `nn.Linear(hidden_dim,1)` — maximal transparency.
- `MLPHead`: stack of Linear + activation (+ optional dropout) ending in scalar output.

---

### `train_model`

```python
from gats import train_model
train_model(
    model, X_train, y_train,
    epochs=100,
    lr=1e-3,
    device="cpu",          # or "cuda"
    weight_decay=0.0       # applied to non (W_Q/W_K/W_V) params
)
```

> Internally uses `_build_optimizer_for_ga_model` (AdamW) with **grouped decay**:  
> **W**<sub>*Q*</sub>/**W**<sub>*K*</sub>: `1e-3`, **W**<sub>*V*</sub>: `1e-4`, others: `weight_decay`.

---

### Analysis utilities

- `generate_comprehensive_results_analysis`: Creates all 7 paper figures
- `analyze_query_attribution` + `visualize_query_analysis`: Deep-dive analysis for specific quarters
- `generate_comprehensive_heatmaps`: Component evolution visualizations

---

## Reproducing Paper Figures (1–7)

After training, call:

```python
from gats import generate_comprehensive_results_analysis
results = generate_comprehensive_results_analysis(
    trained_model=model,
    X_tensor=X_full,               # (1, T, 4) standardized
    y_tensor=y_full,               # (1, T-lookback)
    data=df,                       # original dataframe with 'quarter'
    feature_columns=feature_cols,
    lookback=8,
    save_dir="./ccci_ga_analysis/results_analysis/"
)
```

This writes:

1. `figure1_historical_fit.png`  
2. `figure2_context_trajectory.png`  
3. `figure3_component_evolution.png`  
4. `figure4_attention_heatmap.png`  
5. `figure5_attention_distribution.png`  
6. `figure6_parameter_magnitudes.png`  
7. `figure7_variable_contributions.png`

…and returns a rich dict with all arrays needed to recreate the plots.

---

## Interpretability Outputs

From `complete_ccci_ga_workflow` or manual calls you get:

- **Attention weights** over lags for each *t*: interpret historical precedents.
- **Component magnitudes** (scalar / vectors / bivectors) through time.
- **Query analysis** per specific quarter:
  - Feature-attribution by lag
  - Lag-attribution totals
  - Bivector (interaction) values
  - Attention entropy & focus mode

Example:

```python
from gats import analyze_query_attribution, visualize_query_analysis

qa = analyze_query_attribution(
    trained_model=model,
    X_tensor=X_full,
    query_idx=120,                      # index relative to (lookback .. end)
    feature_columns=feature_cols,
    lookback=8
)
visualize_query_analysis(qa, feature_cols, quarter_label="2015Q4",
                         actual_value=0.85, save_path="./query_2015Q4.png")
```

---

## Tips & Troubleshooting

- **Scaling**: `prepare_ga_sequence_from_df` standardizes features on the **training** sample; keep the `scaler` for out-of-sample use.
- **Lookback mismatch**: Preds start at `t=lookback`. Align targets accordingly (`y_full = raw[lookback:lookback+T_pred]`).
- **Overfitting**: Prefer `head_type="linear"`, reduce `hidden_dim`, or increase **W**<sub>*Q*</sub>/**W**<sub>*K*</sub> weight decay.
- **Sparse bivectors**: Pass a subset of `bivector_planes` for simpler, more explainable interactions.
- **GPU**: Set `device="cuda"` in training for speed.

---

## Citations

**Reference Paper:**  
Agus Sudjianto & Sandi Setiawan (2025). *Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis.*  
Available at SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5441797](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5441797)

> This repository provides the full code implementation and reproducibility package for the paper, including training pipeline, interpretability analysis, and generation of Figures 1–7.

---

**Module file:** `gats.py`  
**Acronym:** *Geometric Algebra Transformer System