# Geometric Algebra Transformer Systems (GATS)

**GATS** (`gats.py`) implements a multivector-based **Geometric Algebra (GA)** representation coupled with **linear attention** for explanatory analysis of consumer credit cycles.  
It is designed to **explain** the dynamics behind credit charge-offs, not just predict them.

> Paper: *Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear‑Attention Framework for Explanatory Economic Analysis* (code and figures reproduced by this repo).

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
  - Heads: [`LinearHead`] and [`MLPHead`](#heads-linearhead-and-mlphead)
  - Training: [`train_model`](#train_model)
  - Analysis utilities
    - `generate_comprehensive_results_analysis`
    - `analyze_query_attribution` + `visualize_query_analysis`
    - `generate_comprehensive_heatmaps`
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

Let the 4-D standardized macro state at time \(t\) be \(X_t = [u_t, s_t, r_t, v_t]\)  
(unemployment, saving rate, PCE growth, revolving credit growth).

### GA embedding
We form a multivector
\[
M_t=\underbrace{\alpha_0}_{\text{scalar}}
+ \sum_{i=1}^4 \alpha_i e_i
+ \sum_{(i,j)\in\Pi} \gamma_{ij}(x_{t,i}-x_{t,j})(e_i\wedge e_j),
\]
with planes \(\Pi=\{(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)\}\).

### Linear attention (feature map \(\phi(x)=\text{LeakyReLU}(x)+1\))
\[
Q_t=\phi(W_Q M_t),\quad K_t=\phi(W_K M_t),\quad V_t=W_V M_t,
\]
over a lookback window \(\mathcal{W}_t=\{t-L,\ldots,t-1\}\).
Denote
\[
S_t=\sum_{\tau\in\mathcal{W}_t} K_\tau V_\tau^\top,\qquad
Z_t=\sum_{\tau\in\mathcal{W}_t} K_\tau.
\]
Then the normalized context is
\[
O_t=\frac{Q_t^\top S_t}{Q_t^\top Z_t + \varepsilon},
\qquad
w_{\tau|t}=\frac{Q_t^\top K_\tau}{\sum_{\kappa\in\mathcal{W}_t} Q_t^\top K_\kappa}.
\]

### Heads
- **Linear**: \(\hat y_t=w_\text{out}^\top O_t + b\).
- **MLP**: \(\hat y_t=W_2\,\sigma(W_1 O_t + b_1)+b_2\) (ReLU/GELU).

> Economic note: the **dot product** \(Q_t^\top K_\tau\) is a scalar **geometric similarity**
between the current multivector state and historical states (suitable for weighting).

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

- `feature_columns`: order matters (maps to GA basis \(e_1..e_4\)).
- `target_column`: typically `CORCACBS`.
- `lookback`: attention window \(L\) (8–12 is typical).
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

- Projections: `W_Q`, `W_K`, `W_V` (learnable `nn.Linear`).
- Feature map: \(\phi(x)=\text{LeakyReLU}(x)+1\) keeps entries positive to stabilize normalization.
- Returns `(contexts, attention_weights)` where `attention_weights[:, t, :]` are the normalized \(w_{\tau|t}\).

**Structured regularization (built into optimizer factory)**  
We apply stronger weight decay to `W_Q`/`W_K` than to `W_V` to stabilize similarity while allowing informative values.

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
> `W_Q/W_K`: `1e-3`, `W_V`: `1e-4`, others: `weight_decay`.

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

- **Attention weights** over lags for each \(t\): interpret historical precedents.
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
- **Overfitting**: Prefer `head_type="linear"`, reduce `hidden_dim`, or increase `W_Q/W_K` weight decay.
- **Sparse bivectors**: Pass a subset of `bivector_planes` for simpler, more explainable interactions.
- **GPU**: Set `device="cuda"` in training for speed.

---

## Citations

- Bahdanau et al. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate.*  
- Vaswani et al. (2017). *Attention Is All You Need.*  
- Choromanski et al. (2020). *Rethinking Attention with Performers.*  
- Hestenes (1984). *New Foundations for Classical Mechanics.*  
- Ghysels et al. (2006, 2007). *MIDAS regressions.*

---

**Module file:** `gats.py`  
**Acronym:** *Geometric Algebra Transformer Systems*
