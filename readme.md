# Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework

## Overview

This repository implements the framework described in "Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis" by Sudjianto & Setiawan (2025). The framework combines Geometric Algebra (GA) with Linear Attention mechanisms to analyze the time-varying geometric relationships in economic data, particularly focusing on consumer credit cycles.

Unlike traditional econometric models that provide global parameter estimates, our approach captures local, time-varying relationships between economic variables through geometric transformations, revealing mechanistically different crisis patterns even when their linear correlations appear similar.

## Key Innovation

The framework addresses a fundamental limitation in economic time series analysis: traditional methods treat magnitude and phase relationships sequentially, first estimating correlations, then separately analyzing temporal patterns. This assumption breaks down during crisis periods when both magnitude and timing relationships evolve simultaneously.

Our approach unifies these relationships through geometric algebra:
- **Inner products** capture projection relationships (correlation-like)  
- **Bivectors** capture rotational dynamics (feedback patterns and spirals)
- **Linear attention** identifies which historical periods are geometrically similar to current conditions

## Mathematical Foundation

### Geometric Algebra Embedding

Economic variables at time t are embedded as multivectors in G(4,0):

```
Xt = ut*e1 + st*e2 + rt*e3 + vt*e4
```

where:
- ut: unemployment rate
- st: savings rate  
- rt: consumption growth
- vt: revolving credit growth

The multivector representation includes:

```
Mt = α0 + Σ(αi*ei) + Σ(γij*(xi,t - xj,t)*(ei ∧ ej))
```

**Components:**
- **Scalar (α0)**: Baseline trend/intercept
- **Vectors (αi*ei)**: Individual variable effects
- **Bivectors (γij*(ei ∧ ej))**: Pairwise variable interactions and feedback patterns

### Linear Attention Mechanism

For each time t with lookback window L:

```
Qt = φ(WQ*Mt), Kt = φ(WK*Mt), Vt = WV*Mt
```

where φ(x) = LeakyReLU(x) + 1 ensures positivity with asymmetric economic response modeling.

The context vector is computed as:

```
Ot = (Qt^T * St) / (Qt^T * Zt + ε)
```

with sufficient statistics:
```
St = Σ(Ki * Vi^T)  [from i=t-L to t-1]
Zt = Σ(Ki)         [from i=t-L to t-1]  
```

### Economic Interpretation as Time-Varying Regression

The framework can be viewed as:

```
yt = β0,t + Σ(βi,t * xi,t) + Σ(γij,t * (xi,t - xj,t)) + εt
```

where coefficients evolve based on geometric similarity to historical periods:

```
βi,t = Σ(wj,t * βi,j)  [attention-weighted historical coefficients]
```

## Architecture Components

### 1. GeometricAlgebraEmbedding
- Maps 4D economic variables to multivector space
- Learnable parameters for scalar bias, vector scaling, and bivector weights
- Captures both individual variable effects and pairwise interactions

### 2. LinearAttention  
- Efficient O(n) attention mechanism with economic asymmetry modeling
- Identifies which historical periods are most relevant for current predictions
- Provides interpretable attention weights for temporal analysis

### 3. Prediction Heads
- **LinearHead**: Simple linear projection for interpretability
- **MLPHead**: Multi-layer perceptron for enhanced predictive performance

### 4. Interpretability Framework
- **Embedding-level occlusion**: Zero out component types to measure impact
- **Query-side occlusion**: Mask attention patterns to understand focus
- **Attribution analysis**: Feature and lag-wise contribution decomposition

## Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd ccci-ga-analysis

# Install dependencies
pip install torch pandas numpy matplotlib seaborn scikit-learn

# Ensure you have CCCI data file
# Data should include: quarter, UNRATE, PSAVERT, PCE_QoQ, REVOLSL_QoQ, CORCACBS
```

## Quick Start

### Basic Usage

```python
from complete_ccci_system import complete_ccci_ga_workflow

# Run complete analysis with linear head
results = complete_ccci_ga_workflow(
    csv_file='ccci_data.csv',
    epochs=100,
    head_type='linear'
)

# Access results
trained_model = results['model']
interpretability = results['interpretability']
query_analyses = results['query_analyses']
insights = results['insights']
```

### Advanced Analysis

```python
# Analyze specific crisis periods with MLP head
crisis_queries = [45, 67, 89, 102]  # Your crisis period indices

results = complete_ccci_ga_workflow(
    csv_file='ccci_data.csv',
    epochs=150,
    head_type='mlp',
    specific_queries=crisis_queries,
    output_dir='./crisis_analysis/'
)

# Deep dive on individual query
from complete_ccci_system import analyze_query_attribution

query_analysis = analyze_query_attribution(
    trained_model, X_tensor, query_idx=67, 
    feature_columns=["UNRATE", "PSAVERT", "PCE_QoQ", "REVOLSL_QoQ"]
)
```

## Experimental Features

### 1. Crisis Pattern Comparison

```python
# Compare geometric signatures across different crises
crisis_periods = {
    '2008_Crisis': [65, 70, 75],  # Crisis period indices
    'COVID_Crisis': [158, 162, 166]
}

for crisis_name, indices in crisis_periods.items():
    for idx in indices:
        query_analysis = analyze_query_attribution(model, X_tensor, idx, features)
        # Analyze bivector patterns, attention focus, component dominance
```

### 2. Component Stability Analysis

```python
from complete_ccci_system import analyze_component_stability

stability_analysis = analyze_component_stability(
    trained_model, X_tensor, feature_columns, 
    lookback=8, window_size=4
)

# Analyze which GA components are most/least stable over time
print("Stability ranking:", stability_analysis['stability_ranking'])
```

### 3. Interactive Query Exploration

```python
from complete_ccci_system import create_interactive_query_explorer

# Generate HTML interface for exploring any quarter
create_interactive_query_explorer(
    trained_model, X_tensor, y_tensor, data, feature_columns,
    output_dir="./interactive_analysis/"
)
```

## Understanding the Output

### 1. Component Analysis
- **Scalar Impact**: How much baseline trends affect predictions
- **Vector Impact**: Individual variable importance  
- **Bivector Impact**: Variable interaction strength (feedback patterns)

### 2. Attention Patterns
- **Attention Weights**: Which historical periods receive focus
- **Attention Entropy**: Whether attention is focused or distributed
- **Most Attended Lag**: Primary temporal focus (e.g., t-3 quarters back)

### 3. Query-Specific Analysis
For each analyzed time period:
- **Prediction vs Actual**: Model accuracy for that quarter
- **Component Breakdown**: Scalar, vector, bivector contributions
- **Feature Attribution**: Which variables at which lags mattered most
- **Geometric Interpretation**: Crisis mechanism (feedback spiral vs sequential)

### 4. Crisis Signatures
Different crises show distinct geometric patterns:
- **2008 Crisis**: High bivector coefficients (γ14 ≈ 0.34) indicating unemployment-credit feedback spirals
- **1990-91 Recession**: Lower bivector coefficients (γ14 ≈ 0.12) showing sequential relationships  
- **COVID Crisis**: Different attention patterns due to policy interventions

## File Structure

```
ccci-ga-analysis/
├── complete_ccci_system.py          # Main analysis framework
├── ccci_data.csv                    # Economic data (FRED sources)
├── README.md                        # This file
├── ccci_ga_analysis/               # Output directory
│   ├── query_1_2008q3_analysis.png # Individual query visualizations
│   ├── query_2_2020q2_analysis.png # COVID period analysis
│   ├── ccci_ga_overview.png        # 6-panel model summary
│   └── ccci_ga_report.md           # Comprehensive analysis report
└── interactive_analysis/           # Interactive exploration tools
    └── query_explorer.html         # Web-based query interface
```

## Data Requirements

Your CSV file should contain quarterly U.S. macroeconomic data with columns:
- **quarter**: Date in format "1980Q1", "1980Q2", etc.
- **UNRATE**: Unemployment rate (%)
- **PSAVERT**: Personal saving rate (%)  
- **PCE_QoQ**: Personal consumption expenditure quarter-over-quarter growth (%)
- **REVOLSL_QoQ**: Revolving credit quarter-over-quarter growth (%)
- **CORCACBS**: Charge-off rate on consumer loans (%) [target variable]

Data sources: Federal Reserve Economic Data (FRED)

## Key Research Findings

1. **Crisis Differentiation**: The framework distinguishes between mechanistically different crises even when correlations are similar:
   - 2008: High bivector coefficients indicating feedback spirals
   - 1990-91: Lower bivector coefficients showing sequential dynamics

2. **Attention Patterns**: Different crises focus on different historical precedents:
   - 2008 crisis: High attention on 2006-2007 precedents
   - 1990-91 recession: Distributed attention on mid-1980s periods

3. **Component Evolution**: Bivector relationships vary significantly across economic cycles while correlations remain stable

4. **Geometric Invariance**: Model learns fundamental economic relationships independent of variable scaling or representation

## Theoretical Guarantees

1. **Geometric Invariance**: Attention scores invariant to isometric transformations
2. **Boundedness**: Predictions remain bounded given input constraints  
3. **Lipschitz Continuity**: Small input changes yield small prediction changes
4. **Exact Impulse Response**: Precise formulas for counterfactual analysis

## Limitations

- Assumes economic relationships have meaningful rotational structure
- Limited sample size for crisis periods may affect robustness
- Computational complexity higher than simple regression (though still linear)
- Framework designed for explanatory analysis, not state-of-the-art prediction

## Extensions & Future Work

1. **Multi-frequency Applications**: Apply quarterly-trained models to monthly data
2. **Additional Economic Variables**: Extend to include financial market indicators
3. **Real-time Applications**: Adapt for nowcasting and early warning systems
4. **Cross-country Analysis**: Compare geometric patterns across different economies

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{sudjianto2025geometric,
  title={Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis},
  author={Sudjianto, Agus and Setiawan, Sandi},
  journal={Submitted},
  year={2025}
}
```

## Contributing

We welcome contributions! Areas of particular interest:
- Additional economic interpretations of geometric relationships
- Extensions to other macroeconomic phenomena  
- Computational optimizations
- Enhanced visualization tools

## License

[Specify your license here]

## Contact

- Agus Sudjianto: agus.sudjianto@h2o.ai
- Sandi Setiawan: sandi.setiawan@cantab.net

---

*This framework provides a novel lens for understanding economic dynamics through geometric algebra, moving beyond traditional correlation-based approaches to capture the rotational relationships and feedback patterns that distinguish different crisis mechanisms.*