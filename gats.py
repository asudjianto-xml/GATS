"""
Complete CCCI GA Analysis System
Combines the GA model architecture with enhanced interpretability analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Union, Any
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GA MODEL ARCHITECTURE (from your provided code)
# ============================================================

class GeometricAlgebraEmbedding(nn.Module):
    """GA-inspired embedding for 4D economic variables"""
    def __init__(self, bivector_planes: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.bivector_planes = bivector_planes or [(0,1),(0, 2), (0, 3), (1,2), (1,3), (2,3)]
        self.n_vars = 4
        self.n_biv = len(self.bivector_planes)
        self.multivector_dim = 1 + self.n_vars + self.n_biv

        self.scalar_bias = nn.Parameter(torch.zeros(1))
        self.vector_scale = nn.Parameter(torch.ones(self.n_vars))
        self.vector_bias = nn.Parameter(torch.zeros(self.n_vars))
        self.bivector_weight = nn.Parameter(torch.ones(self.n_biv))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        out = torch.zeros(B, L, self.multivector_dim, device=x.device, dtype=x.dtype)
        
        out[..., 0] = 1.0 + self.scalar_bias
        scale = self.vector_scale.view(1, 1, -1)
        bias  = self.vector_bias.view(1, 1, -1)
        out[..., 1:5] = x * scale + bias
        
        base = 1 + self.n_vars
        for k, (i, j) in enumerate(self.bivector_planes):
            out[..., base + k] = self.bivector_weight[k] * (x[..., i] - x[..., j])
        
        return out

class LinearAttention(nn.Module):
    """Linear attention with phi(x) = LeakyReLU(x)+1.0"""
    def __init__(self, multivector_dim: int, hidden_dim: int = 32, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.W_Q = nn.Linear(multivector_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(multivector_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(multivector_dim, hidden_dim, bias=False)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=0.01) + 1.0

    def forward(self, M: torch.Tensor, lookback: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = M.shape
        Q = self.phi(self.W_Q(M))
        K = self.phi(self.W_K(M))
        V = self.W_V(M)

        contexts = []
        weights_all = []

        for t in range(lookback, L):
            q_t = Q[:, t, :]
            k_win = K[:, t - lookback: t, :]
            v_win = V[:, t - lookback: t, :]

            S = torch.einsum('blh,blg->bhg', k_win, v_win)
            Z = k_win.sum(dim=1)
            num = torch.einsum('bh,bhg->bg', q_t, S)
            den = (q_t * Z).sum(dim=-1, keepdim=True) + self.eps
            O_t = num / den
            contexts.append(O_t.unsqueeze(1))

            scores = torch.einsum('bh,blh->bl', q_t, k_win)
            weights = scores / (scores.sum(dim=-1, keepdim=True) + self.eps)
            weights_all.append(weights.unsqueeze(1))

        contexts = torch.cat(contexts, dim=1)
        attention_weights = torch.cat(weights_all, dim=1)
        return contexts, attention_weights

class LinearHead(nn.Module):
    """Linear prediction head"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        y = self.proj(contexts)
        return y.squeeze(-1)

class MLPHead(nn.Module):
    """MLP prediction head"""
    def __init__(self, hidden_dim: int,
                 hidden_dims: Tuple[int, ...] = (64, 32),
                 dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        layers = []
        last = hidden_dim
        Act = nn.ReLU if activation.lower() == "relu" else nn.GELU
        for h in hidden_dims:
            layers += [nn.Linear(last, h), Act()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, 1))
        self.ff = nn.Sequential(*layers)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        B, T, H = contexts.shape
        flat = contexts.reshape(B * T, H)
        y = self.ff(flat)
        return y.view(B, T, 1).squeeze(-1)

class GALinearAttentionModel(nn.Module):
    """Complete GA model with interpretability"""
    def __init__(self,
                 bivector_planes: Optional[List[Tuple[int, int]]] = None,
                 hidden_dim: int = 32,
                 lookback: int = 8,
                 eps: float = 1e-6,
                 include_query_side: bool = True,
                 head_type: str = "linear",
                 mlp_hidden: Tuple[int, ...] = (64, 32),
                 mlp_dropout: float = 0.0,
                 mlp_activation: str = "relu"):
        super().__init__()
        self.ga_embedding = GeometricAlgebraEmbedding(bivector_planes)
        self.linear_attention = LinearAttention(self.ga_embedding.multivector_dim,
                                                hidden_dim, eps=eps)
        self.lookback = lookback
        self.include_query_side = include_query_side
        self.hidden_dim = hidden_dim

        if head_type.lower() == "mlp":
            self.head = MLPHead(hidden_dim, mlp_hidden, mlp_dropout, mlp_activation)
            self.head_type = "mlp"
        else:
            self.head = LinearHead(hidden_dim)
            self.head_type = "linear"

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        M = self.ga_embedding(x)
        contexts, attention_weights = self.linear_attention(M, self.lookback)
        predictions = self.head(contexts)
        interpretability = self._compute_interpretability(M, contexts, predictions)
        return predictions, attention_weights, interpretability

    def _delta_y_query_only(self, M: torch.Tensor, block_slice: slice) -> float:
        """Query-side occlusion analysis"""
        with torch.no_grad():
            Q = self.linear_attention.phi(self.linear_attention.W_Q(M))
            K = self.linear_attention.phi(self.linear_attention.W_K(M))
            V = self.linear_attention.W_V(M)

            t = M.size(1) - 1
            Lb = self.lookback
            k_win = K[:, t - Lb: t, :]
            v_win = V[:, t - Lb: t, :]
            S = torch.einsum('blh,blg->bhg', k_win, v_win)
            Z = k_win.sum(dim=1)
            q = Q[:, t, :]

            num = torch.einsum('bh,bhg->bg', q, S)
            den = (q * Z).sum(dim=-1, keepdim=True) + self.linear_attention.eps
            O = num / den
            y = self.head(O.unsqueeze(1)).squeeze(1)

            WQ = self.linear_attention.W_Q.weight.detach()
            Dm = M.size(-1)
            mask = torch.zeros(Dm, device=M.device)
            mask[block_slice] = 1.0
            h_imp = (WQ.abs() * mask.view(1, -1)).sum(dim=1)
            h_mask = (h_imp / (h_imp.max() + 1e-12)).view(1, -1)
            q_masked = q * (1.0 - h_mask)

            num_b = torch.einsum('bh,bhg->bg', q_masked, S)
            den_b = (q_masked * Z).sum(dim=-1, keepdim=True) + self.linear_attention.eps
            O_b = num_b / den_b
            y_b = self.head(O_b.unsqueeze(1)).squeeze(1)
            return (y - y_b).abs().mean().item()

    def _compute_interpretability(self, M, contexts, predictions) -> Dict[str, float]:
        """Compute interpretability metrics"""
        B, L, Dm = M.shape
        
        # Magnitudes
        scalar_mag = M[..., 0].abs().mean().item()
        vector_mag = M[..., 1:5].abs().mean().item()
        biv_mag = M[..., 5:].abs().mean().item()

        # Embedding-level occlusions
        with torch.no_grad():
            base_preds = predictions

            # No scalar
            M0 = M.clone(); M0[..., 0] = 0
            ctx0, _ = self.linear_attention(M0, self.lookback)
            preds0 = self.head(ctx0)
            delta_scalar = (base_preds - preds0).abs().mean().item()

            # No vectors
            Mv = M.clone(); Mv[..., 1:5] = 0
            ctxv, _ = self.linear_attention(Mv, self.lookback)
            predsv = self.head(ctxv)
            delta_vectors = (base_preds - predsv).abs().mean().item()

            # No bivectors
            Mb = M.clone(); Mb[..., 5:] = 0
            ctxb, _ = self.linear_attention(Mb, self.lookback)
            predsb = self.head(ctxb)
            delta_bivectors = (base_preds - predsb).abs().mean().item()

        out = {
            'scalar_magnitude': scalar_mag,
            'vector_magnitude': vector_mag,
            'bivector_magnitude': biv_mag,
            'scalar_causal_impact': delta_scalar,
            'vector_causal_impact': delta_vectors,
            'bivector_causal_impact': delta_bivectors,
        }

        if self.include_query_side:
            out['scalar_causal_impact_Qonly'] = self._delta_y_query_only(M, slice(0, 1))
            out['vector_causal_impact_Qonly'] = self._delta_y_query_only(M, slice(1, 1+4))
            out['bivector_causal_impact_Qonly'] = self._delta_y_query_only(M, slice(1+4, Dm))

        return out

# ============================================================
# ENHANCED INTERPRETABILITY ANALYSIS
# ============================================================

def analyze_query_attribution(
    trained_model: GALinearAttentionModel, 
    X_tensor: torch.Tensor, 
    query_idx: int, 
    feature_columns: List[str],
    lookback: int = 8,
    device: str = None
) -> Dict[str, Any]:
    """Detailed attribution analysis for a specific query"""
    if device is None:
        device = next(trained_model.parameters()).device
    
    trained_model.eval()
    
    # Get base prediction and components
    with torch.no_grad():
        X_query = X_tensor.to(device)
        base_pred, base_attention, base_interp = trained_model(X_query)
        base_multivectors = trained_model.ga_embedding(X_query)
    
    query_pred = base_pred.squeeze(0)[query_idx].cpu().item()
    query_attention = base_attention.squeeze(0)[query_idx].cpu()
    query_multivector = base_multivectors.squeeze(0)[query_idx + lookback].cpu()
    
    # Feature-wise attribution (for each input feature at each lag)
    feature_attributions = np.zeros((len(feature_columns), lookback))
    
    for feat_idx in range(len(feature_columns)):
        for lag in range(lookback):
            # Create modified input with this feature at this lag set to zero
            X_modified = X_tensor.clone()
            X_modified[0, query_idx + lookback - lag - 1, feat_idx] = 0
            
            with torch.no_grad():
                modified_pred, _, _ = trained_model(X_modified.to(device))
                modified_pred_val = modified_pred.squeeze(0)[query_idx].cpu().item()
                feature_attributions[feat_idx, lag] = query_pred - modified_pred_val
    
    # Lag-wise total attribution
    lag_attributions = np.zeros(lookback)
    for lag in range(lookback):
        X_modified = X_tensor.clone()
        X_modified[0, query_idx + lookback - lag - 1, :] = 0
        
        with torch.no_grad():
            modified_pred, _, _ = trained_model(X_modified.to(device))
            modified_pred_val = modified_pred.squeeze(0)[query_idx].cpu().item()
            lag_attributions[lag] = query_pred - modified_pred_val
    
    # Component-wise analysis from multivector
    scalar_val = query_multivector[0].item()
    vector_vals = query_multivector[1:5].numpy()
    bivector_vals = query_multivector[5:].numpy()
    
    # Create bivector pairs
    bivector_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    bivector_analysis = {}
    for i, (feat_i, feat_j) in enumerate(bivector_pairs[:len(bivector_vals)]):
        pair_name = f'{feature_columns[feat_i]}∧{feature_columns[feat_j]}'
        bivector_analysis[pair_name] = bivector_vals[i]
    
    return {
        'query_idx': query_idx,
        'prediction': query_pred,
        'scalar_value': scalar_val,
        'vector_values': dict(zip(feature_columns, vector_vals)),
        'bivector_values': bivector_analysis,
        'attention_weights': query_attention.numpy(),
        'feature_attributions': feature_attributions,
        'lag_attributions': lag_attributions,
        'attention_entropy': -(query_attention * torch.log(query_attention + 1e-8)).sum().item()
    }

def visualize_query_analysis(
    query_analysis: Dict[str, Any], 
    feature_columns: List[str],
    quarter_label: str,
    actual_value: float = None,
    save_path: str = None
) -> None:
    """Create comprehensive visualization for a specific query"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Component Values
    ax1 = plt.subplot(3, 4, 1)
    components = ['Scalar']
    component_values = [query_analysis['scalar_value']]
    
    for feat, val in query_analysis['vector_values'].items():
        components.append(f'V_{feat}')
        component_values.append(val)
    
    for biv_name, val in query_analysis['bivector_values'].items():
        components.append(f'B_{biv_name}')
        component_values.append(val)
    
    colors = ['red'] + ['blue'] * len(feature_columns) + ['green'] * len(query_analysis['bivector_values'])
    bars = ax1.barh(range(len(components)), component_values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(components)))
    ax1.set_yticklabels(components, fontsize=8)
    ax1.set_title(f'GA Components - {quarter_label}')
    ax1.set_xlabel('Component Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. Attention Weights
    ax2 = plt.subplot(3, 4, 2)
    lags = [f't-{i+1}' for i in range(len(query_analysis['attention_weights']))]
    bars = ax2.bar(range(len(lags)), query_analysis['attention_weights'])
    ax2.set_xticks(range(len(lags)))
    ax2.set_xticklabels(lags, rotation=45)
    ax2.set_title(f'Attention Weights - {quarter_label}')
    ax2.set_ylabel('Attention Weight')
    ax2.grid(True, alpha=0.3)
    
    max_idx = np.argmax(query_analysis['attention_weights'])
    bars[max_idx].set_color('red')
    
    # 3. Feature Attribution Heatmap
    ax3 = plt.subplot(3, 4, (3, 4))
    im = ax3.imshow(query_analysis['feature_attributions'], cmap='RdBu_r', aspect='auto')
    ax3.set_title(f'Feature Attribution by Lag - {quarter_label}')
    ax3.set_xlabel('Lookback Period')
    ax3.set_ylabel('Features')
    ax3.set_xticks(range(len(lags)))
    ax3.set_xticklabels(lags)
    ax3.set_yticks(range(len(feature_columns)))
    ax3.set_yticklabels(feature_columns)
    plt.colorbar(im, ax=ax3, label='Attribution')
    
    # Add significant values as text
    for i in range(len(feature_columns)):
        for j in range(len(lags)):
            val = query_analysis['feature_attributions'][i, j]
            if abs(val) > np.std(query_analysis['feature_attributions']) * 1.5:
                ax3.text(j, i, f'{val:.2f}', ha='center', va='center', 
                        color='white' if abs(val) > np.max(np.abs(query_analysis['feature_attributions'])) * 0.5 else 'black',
                        fontweight='bold', fontsize=8)
    
    # 4. Lag Attribution
    ax4 = plt.subplot(3, 4, 5)
    bars = ax4.bar(range(len(lags)), query_analysis['lag_attributions'])
    ax4.set_xticks(range(len(lags)))
    ax4.set_xticklabels(lags, rotation=45)
    ax4.set_title(f'Total Lag Attribution - {quarter_label}')
    ax4.set_ylabel('Attribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Component Magnitude Comparison
    ax5 = plt.subplot(3, 4, 6)
    scalar_mag = abs(query_analysis['scalar_value'])
    vector_mag = np.mean([abs(v) for v in query_analysis['vector_values'].values()])
    bivector_mag = np.mean([abs(v) for v in query_analysis['bivector_values'].values()])
    
    comp_types = ['Scalar', 'Vector\n(avg)', 'Bivector\n(avg)']
    comp_mags = [scalar_mag, vector_mag, bivector_mag]
    colors = ['red', 'blue', 'green']
    
    bars = ax5.bar(comp_types, comp_mags, color=colors, alpha=0.7)
    ax5.set_title(f'Component Magnitudes - {quarter_label}')
    ax5.set_ylabel('Absolute Magnitude')
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary Text
    ax6 = plt.subplot(3, 4, (7, 12))
    ax6.axis('off')
    
    summary_text = f"""Query Analysis Summary for {quarter_label}
═══════════════════════════════════════════════

Prediction: {query_analysis['prediction']:.3f}%"""
    
    if actual_value is not None:
        summary_text += f"""
Actual: {actual_value:.3f}%
Error: {abs(query_analysis['prediction'] - actual_value):.3f}%"""
    
    summary_text += f"""

ATTENTION ANALYSIS:
• Primary focus: t-{np.argmax(query_analysis['attention_weights'])+1}
• Attention entropy: {query_analysis['attention_entropy']:.3f}
• Distribution: {'Focused' if query_analysis['attention_entropy'] < 1.5 else 'Distributed'}

GA COMPONENT ANALYSIS:
• Scalar: {query_analysis['scalar_value']:.3f}
• Strongest vector: {max(query_analysis['vector_values'].items(), key=lambda x: abs(x[1]))[0]} = {max(query_analysis['vector_values'].values(), key=abs):.3f}
• Strongest bivector: {max(query_analysis['bivector_values'].items(), key=lambda x: abs(x[1]))[0]} = {max(query_analysis['bivector_values'].values(), key=abs):.3f}

FEATURE ATTRIBUTION:
• Most impactful feature: {feature_columns[np.argmax(np.sum(np.abs(query_analysis['feature_attributions']), axis=1))]}
• Most impactful lag: t-{np.argmax(np.abs(query_analysis['lag_attributions']))+1}
• Total attribution: {np.sum(np.abs(query_analysis['lag_attributions'])):.3f}

INTERPRETATION:
• Component dominance: {'Scalar' if scalar_mag > max(vector_mag, bivector_mag) else 'Vector' if vector_mag > bivector_mag else 'Bivector'}
• Temporal focus: {'Recent' if np.argmax(query_analysis['attention_weights']) < 2 else 'Medium-term' if np.argmax(query_analysis['attention_weights']) < 5 else 'Long-term'}
• Interaction complexity: {'High' if bivector_mag > vector_mag else 'Moderate' if bivector_mag > 0.5 * vector_mag else 'Low'}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Query analysis saved: {save_path}")
    
    plt.show()

# ============================================================
# COMPLETE WORKFLOW INTEGRATION
# ============================================================

def complete_ccci_ga_workflow(
    csv_file: str = 'ccci_data.csv',
    feature_columns: List[str] = None,
    target_column: str = 'CORCACBS',
    lookback: int = 8,
    train_frac: float = 0.8,
    epochs: int = 100,
    specific_queries: List[int] = None,
    head_type: str = "linear",  # "linear" or "mlp"
    output_dir: str = "./ccci_ga_analysis/"
) -> Dict[str, Any]:
    """
    Complete end-to-end CCCI analysis workflow
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if feature_columns is None:
        feature_columns = ["UNRATE", "PSAVERT", "PCE_QoQ", "REVOLSL_QoQ"]
    
    print("="*80)
    print("COMPLETE CCCI GA ANALYSIS WORKFLOW")
    print("="*80)
    
    # 1. Load and prepare data
    print("\n1. LOADING AND PREPARING DATA...")
    data = pd.read_csv(csv_file)
    data.columns = [col.strip().replace('\r', '') for col in data.columns]
    
    print(f"Data shape: {data.shape}")
    print(f"Features: {feature_columns}")
    print(f"Target: {target_column}")
    print(f"Date range: {data['quarter'].iloc[0]} to {data['quarter'].iloc[-1]}")
    
    # Prepare sequence data
    def prepare_ga_sequence_from_df(df, feature_cols, target_col, lookback, train_frac):
        scaler = StandardScaler()
        n = len(df)
        split_idx = int(np.floor(train_frac * n))
        split_idx = max(split_idx, lookback + 1)
        split_idx = min(split_idx, n - (lookback + 1))
        
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        X_train_raw = df_train[feature_cols].values.astype(np.float32)
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test_raw = df_test[feature_cols].values.astype(np.float32)
        X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
        
        y_train_full = df_train[target_col].values.astype(np.float32)
        y_test_full = df_test[target_col].values.astype(np.float32)
        y_train = y_train_full[lookback:].copy()
        y_test = y_test_full[lookback:].copy()
        
        X_train = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(0)
        X_test = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(0)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(0)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(0)
        
        return X_train, y_train, X_test, y_test, scaler
    
    X_train, y_train, X_test, y_test, scaler = prepare_ga_sequence_from_df(
        data, feature_columns, target_column, lookback, train_frac
    )
    
    print(f"Training: X{X_train.shape}, y{y_train.shape}")
    print(f"Testing: X{X_test.shape}, y{y_test.shape}")
    
    # 2. Create and train GA model
    print(f"\n2. CREATING AND TRAINING GA MODEL (head_type='{head_type}')...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = GALinearAttentionModel(
        hidden_dim=32,
        lookback=lookback,
        head_type=head_type,
        mlp_hidden=(64, 32) if head_type == "mlp" else None,
        include_query_side=True
    ).to(device)
    
    # Training function
    def train_model(model, X, y, epochs, lr=1e-3):
        model.train()
        X, y = X.to(device), y.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        loss_fn = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            preds, _, _ = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch == 1 or epoch % 20 == 0:
                rmse = float(torch.sqrt(loss).item())
                print(f"  Epoch {epoch:02d} | MSE {loss.item():.4f} | RMSE {rmse:.4f}")
    
    train_model(model, X_train, y_train, epochs)
    
    # 3. Evaluate model
    print("\n3. EVALUATING MODEL...")
    model.eval()
    with torch.no_grad():
        train_preds, train_attention, train_interp = model(X_train.to(device))
        test_preds, test_attention, test_interp = model(X_test.to(device))
        
        train_rmse = torch.sqrt(F.mse_loss(train_preds, y_train.to(device))).item()
        test_rmse = torch.sqrt(F.mse_loss(test_preds, y_test.to(device))).item()
        
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        
        print(f"\nInterpretability Analysis:")
        for key, value in train_interp.items():
            print(f"  {key}: {value:.4f}")
    
    # 4. Prepare full dataset for analysis
    print("\n4. PREPARING FULL DATASET FOR COMPREHENSIVE ANALYSIS...")
    X_full = torch.cat([X_train, X_test], dim=1)
    y_full = torch.cat([y_train, y_test], dim=1)
    
    with torch.no_grad():
        full_preds, full_attention, full_interp = model(X_full.to(device))
        full_multivectors = model.ga_embedding(X_full.to(device))
    
    # Move back to CPU for analysis
    full_preds = full_preds.cpu()
    full_attention = full_attention.cpu()
    full_multivectors = full_multivectors.cpu()
    
    # 5. Auto-select interesting queries if not specified
    if specific_queries is None:
        print("\n5. AUTO-SELECTING INTERESTING QUERIES...")
        specific_queries = []
        
        # High charge-off periods
        high_periods = torch.topk(y_full.squeeze(0), k=4).indices.tolist()
        specific_queries.extend(high_periods)
        
        # Recent periods
        T = y_full.shape[1]
        specific_queries.extend([T-12, T-6, T-2])
        
        # Some distributed periods
        specific_queries.extend([T//4, T//2, 3*T//4])
        
        specific_queries = sorted(list(set(specific_queries)))[:8]
        print(f"Selected queries: {specific_queries}")
    
    # 6. Perform detailed query analysis
    print("\n6. PERFORMING DETAILED QUERY ANALYSIS...")
    query_analyses = {}
    
    for i, query_idx in enumerate(specific_queries):
        if 0 <= query_idx < y_full.shape[1]:
            quarter_label = data['quarter'].iloc[query_idx + lookback]
            actual_value = y_full.squeeze(0)[query_idx].item()
            
            print(f"  Analyzing {quarter_label} (idx {query_idx})...")
            
            query_analysis = analyze_query_attribution(
                model, X_full, query_idx, feature_columns, lookback, device
            )
            
            # Create visualization
            save_path = os.path.join(output_dir, f"query_{i+1}_{quarter_label.replace('Q', 'q')}_analysis.png")
            visualize_query_analysis(
                query_analysis, feature_columns, quarter_label, 
                actual_value, save_path
            )
            
            query_analysis['quarter'] = quarter_label
            query_analysis['actual_value'] = actual_value
            query_analysis['prediction_error'] = abs(query_analysis['prediction'] - actual_value)
            
            query_analyses[f'query_{i+1}_{quarter_label}'] = query_analysis
    
    # 7. Create comprehensive overview
    print("\n7. CREATING COMPREHENSIVE OVERVIEW...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Time series plot
    ax1 = axes[0, 0]
    time_axis = data['quarter_date'].iloc[lookback:lookback+len(y_full.squeeze(0))]
    ax1.plot(time_axis, y_full.squeeze(0).numpy(), 'b-', label='Actual', alpha=0.8)
    ax1.plot(time_axis, full_preds.squeeze(0).numpy(), 'r--', label='Predicted', alpha=0.8)
    ax1.set_title('CCCI Charge-off Rates: Actual vs Predicted')
    ax1.set_ylabel('Charge-off Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add crisis periods
    crisis_periods = {
        '2008_Crisis': ('2007Q4', '2009Q2'),
        'COVID': ('2020Q2', '2020Q4')
    }
    for crisis_name, (start, end) in crisis_periods.items():
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            if start_date >= time_axis.min() and end_date <= time_axis.max():
                ax1.axvspan(start_date, end_date, alpha=0.2, color='red', label=crisis_name if crisis_name == '2008_Crisis' else "")
        except:
            pass
    
    # Component importance
    ax2 = axes[0, 1]
    comp_names = ['Scalar', 'Vector', 'Bivector']
    comp_impacts = [
        full_interp['scalar_causal_impact'],
        full_interp['vector_causal_impact'],
        full_interp['bivector_causal_impact']
    ]
    bars = ax2.bar(comp_names, comp_impacts, color=['red', 'blue', 'green'], alpha=0.7)
    ax2.set_title('GA Component Importance')
    ax2.set_ylabel('Causal Impact')
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, impact in zip(bars, comp_impacts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{impact:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Attention pattern
    ax3 = axes[0, 2]
    avg_attention = full_attention.mean(dim=(0, 1)).numpy()
    lag_labels = [f't-{i+1}' for i in range(len(avg_attention))]
    bars = ax3.bar(range(len(avg_attention)), avg_attention)
    ax3.set_title('Average Attention Pattern')
    ax3.set_xlabel('Lookback Period')
    ax3.set_ylabel('Attention Weight')
    ax3.set_xticks(range(len(lag_labels)))
    ax3.set_xticklabels(lag_labels)
    ax3.grid(True, alpha=0.3)
    
    # Prediction scatter
    ax4 = axes[1, 0]
    actual_vals = y_full.squeeze(0).numpy()
    pred_vals = full_preds.squeeze(0).numpy()
    ax4.scatter(actual_vals, pred_vals, alpha=0.6, color='purple')
    ax4.plot([actual_vals.min(), actual_vals.max()], [actual_vals.min(), actual_vals.max()], 'r--')
    ax4.set_xlabel('Actual Charge-off Rate (%)')
    ax4.set_ylabel('Predicted Charge-off Rate (%)')
    ax4.set_title(f'Predictions vs Actual\n(Test RMSE: {test_rmse:.3f}%)')
    ax4.grid(True, alpha=0.3)
    
    # Query analysis summary
    ax5 = axes[1, 1]
    if query_analyses:
        query_errors = [q['prediction_error'] for q in query_analyses.values()]
        query_names = [q['quarter'] for q in query_analyses.values()]
        bars = ax5.bar(range(len(query_errors)), query_errors)
        ax5.set_title('Query-Specific Prediction Errors')
        ax5.set_ylabel('Prediction Error (%)')
        ax5.set_xticks(range(len(query_names)))
        ax5.set_xticklabels(query_names, rotation=45)
        ax5.grid(True, alpha=0.3)
    
    # Component evolution (simplified)
    ax6 = axes[1, 2]
    scalar_series = full_multivectors.squeeze(0)[lookback:, 0].numpy()
    vector_mag_series = full_multivectors.squeeze(0)[lookback:, 1:5].abs().mean(dim=1).numpy()
    bivector_mag_series = full_multivectors.squeeze(0)[lookback:, 5:].abs().mean(dim=1).numpy()
    
    ax6.plot(time_axis, np.abs(scalar_series), 'r-', label='|Scalar|', alpha=0.7)
    ax6.plot(time_axis, vector_mag_series, 'b-', label='|Vector|', alpha=0.7)
    ax6.plot(time_axis, bivector_mag_series, 'g-', label='|Bivector|', alpha=0.7)
    ax6.set_title('Component Magnitude Evolution')
    ax6.set_ylabel('Magnitude')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    overview_path = os.path.join(output_dir, 'ccci_ga_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Overview saved: {overview_path}")
    
    # 8. Generate insights report
    print("\n8. GENERATING INSIGHTS REPORT...")
    
    def generate_insights(model_results, query_results):
        insights = {}
        
        # Performance insight
        if test_rmse < 0.5:
            insights['performance'] = f"Excellent predictive accuracy (RMSE: {test_rmse:.3f}%), indicating the GA model effectively captures credit cycle dynamics."
        elif test_rmse < 1.0:
            insights['performance'] = f"Good predictive accuracy (RMSE: {test_rmse:.3f}%), showing reliable forecasting capability."
        else:
            insights['performance'] = f"Moderate predictive accuracy (RMSE: {test_rmse:.3f}%), suggesting room for improvement in capturing complex patterns."
        
        # Component insight
        comp_ranking = [
            ('Scalar', full_interp['scalar_causal_impact']),
            ('Vector', full_interp['vector_causal_impact']),
            ('Bivector', full_interp['bivector_causal_impact'])
        ]
        comp_ranking.sort(key=lambda x: x[1], reverse=True)
        
        dominant_component = comp_ranking[0][0]
        if dominant_component == 'Bivector':
            insights['ga_structure'] = "Bivector dominance indicates that interactions between economic variables are more predictive than individual variables, suggesting complex non-linear relationships in credit cycles."
        elif dominant_component == 'Vector':
            insights['ga_structure'] = "Vector dominance shows individual economic variables have strong predictive power, indicating more direct linear relationships in credit behavior."
        else:
            insights['ga_structure'] = "Scalar dominance suggests baseline trends are most important, indicating strong temporal persistence in charge-off patterns."
        
        # Attention insight
        most_attended_lag = np.argmax(avg_attention)
        if most_attended_lag <= 1:
            insights['temporal_focus'] = f"Model primarily focuses on recent quarters (t-{most_attended_lag+1}), suggesting credit conditions respond quickly to economic changes."
        elif most_attended_lag >= 6:
            insights['temporal_focus'] = f"Model focuses on distant history (t-{most_attended_lag+1}), indicating long-term economic cycles drive credit conditions."
        else:
            insights['temporal_focus'] = f"Model balances short and medium-term history (t-{most_attended_lag+1}), capturing both immediate and cyclical patterns."
        
        # Query-specific insights
        if query_results:
            avg_query_error = np.mean([q['prediction_error'] for q in query_results.values()])
            insights['query_performance'] = f"Average query-specific error of {avg_query_error:.3f}% shows {'excellent' if avg_query_error < 0.3 else 'good' if avg_query_error < 0.6 else 'moderate'} individual period accuracy."
        
        return insights
    
    insights = generate_insights(full_interp, query_analyses)
    
    # Save comprehensive report
    report_path = os.path.join(output_dir, 'ccci_ga_report.md')
    with open(report_path, 'w') as f:
        f.write("# CCCI GA Model Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Model Architecture**: GA Linear Attention with {head_type.upper()} head\n")
        f.write(f"- **Training RMSE**: {train_rmse:.4f}%\n")
        f.write(f"- **Testing RMSE**: {test_rmse:.4f}%\n")
        f.write(f"- **Analysis Period**: {data['quarter'].iloc[lookback]} to {data['quarter'].iloc[-1]}\n")
        f.write(f"- **Queries Analyzed**: {len(query_analyses)}\n\n")
        
        f.write("## Key Insights\n\n")
        for insight_type, insight_text in insights.items():
            f.write(f"### {insight_type.replace('_', ' ').title()}\n")
            f.write(f"{insight_text}\n\n")
        
        f.write("## Component Analysis\n\n")
        f.write("### GA Component Importance:\n")
        comp_ranking = [
            ('Scalar', full_interp['scalar_causal_impact']),
            ('Vector', full_interp['vector_causal_impact']),
            ('Bivector', full_interp['bivector_causal_impact'])
        ]
        comp_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (name, impact) in enumerate(comp_ranking, 1):
            f.write(f"{i}. **{name}**: {impact:.4f}\n")
        
        f.write("\n### Component Magnitudes:\n")
        f.write(f"- Scalar: {full_interp['scalar_magnitude']:.4f}\n")
        f.write(f"- Vector: {full_interp['vector_magnitude']:.4f}\n")
        f.write(f"- Bivector: {full_interp['bivector_magnitude']:.4f}\n\n")
        
        if query_analyses:
            f.write("## Query-Specific Analysis\n\n")
            for query_name, analysis in list(query_analyses.items())[:5]:
                f.write(f"### {analysis['quarter']}\n")
                f.write(f"- Prediction: {analysis['prediction']:.3f}% (Actual: {analysis['actual_value']:.3f}%)\n")
                f.write(f"- Error: {analysis['prediction_error']:.3f}%\n")
                f.write(f"- Primary attention: t-{np.argmax(analysis['attention_weights'])+1}\n")
                
                # Find dominant component for this query
                scalar_mag = abs(analysis['scalar_value'])
                vector_mag = np.mean([abs(v) for v in analysis['vector_values'].values()])
                bivector_mag = np.mean([abs(v) for v in analysis['bivector_values'].values()]) if analysis['bivector_values'] else 0
                dominant = 'Scalar' if scalar_mag > max(vector_mag, bivector_mag) else 'Vector' if vector_mag > bivector_mag else 'Bivector'
                f.write(f"- Dominant component: {dominant}\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- Individual query analyses: `query_*_analysis.png`\n")
        f.write("- Model overview: `ccci_ga_overview.png`\n")
        f.write("- This report: `ccci_ga_report.md`\n")
    
    print(f"Report saved: {report_path}")
    
    # 9. Compile final results
    results = {
        'model': model,
        'scaler': scaler,
        'data_info': {
            'train_size': len(X_train.squeeze(0)),
            'test_size': len(X_test.squeeze(0)),
            'features': feature_columns,
            'target': target_column,
            'lookback': lookback
        },
        'performance': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': 1 - F.mse_loss(train_preds, y_train.to(device)) / torch.var(y_train.to(device)),
            'test_r2': 1 - F.mse_loss(test_preds, y_test.to(device)) / torch.var(y_test.to(device))
        },
        'interpretability': full_interp,
        'query_analyses': query_analyses,
        'insights': insights,
        'predictions': {
            'train': train_preds.cpu(),
            'test': test_preds.cpu(),
            'full': full_preds.cpu()
        },
        'attention': {
            'train': train_attention.cpu(),
            'test': test_attention.cpu(),
            'full': full_attention.cpu()
        }
    }
    
    print("\n" + "="*80)
    print("COMPLETE CCCI GA ANALYSIS FINISHED")
    print("="*80)
    print(f"📊 Model Performance:")
    print(f"   Training RMSE: {train_rmse:.4f}%")
    print(f"   Testing RMSE:  {test_rmse:.4f}%")
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📈 Query analyses: {len(query_analyses)} detailed visualizations")
    print(f"🧠 Insights: {len(insights)} key findings")
    print(f"📋 Report: {report_path}")
    
    return results

# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
# Basic usage with linear head
results = complete_ccci_ga_workflow(
    csv_file='ccci_data.csv',
    epochs=100,
    head_type='linear'
)

# Advanced usage with MLP head and specific queries
results = complete_ccci_ga_workflow(
    csv_file='ccci_data.csv',
    epochs=150,
    head_type='mlp',
    specific_queries=[45, 67, 89, 102],  # Crisis periods
    output_dir='./advanced_analysis/'
)

# Access trained model for further analysis
trained_model = results['model']
interpretability = results['interpretability']
query_results = results['query_analyses']

# Use the original enhanced analysis functions for deeper insights
from your_enhanced_analysis_module import analyze_component_stability
stability_analysis = analyze_component_stability(trained_model, X_tensor, feature_columns)
"""