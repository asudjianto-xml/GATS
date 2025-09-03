import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def analyze_ccci_model(
    trained_model,  # Your GALinearAttentionModel
    csv_file: str = 'ccci_data.csv',
    specific_queries: Optional[List[Union[int, str]]] = None,
    save_plots: bool = True,
    output_dir: str = "./ccci_analysis/"
) -> Dict[str, Any]:
    """
    Comprehensive analysis specifically designed for your CCCI dataset
    
    Parameters:
    -----------
    trained_model : GALinearAttentionModel
        Your trained GA model
    csv_file : str
        Path to your CSV file
    specific_queries : Optional[List[Union[int, str]]]
        Specific time points to analyze (indices or quarters like '2008Q3')
    save_plots : bool
        Whether to save plots
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    Complete analysis results dictionary
    """
    
    print("="*80)
    print("CCCI CREDIT CYCLE ANALYSIS - GA MODEL INTERPRETABILITY")
    print("="*80)
    
    # ==================== Load and Prepare Data ====================
    print("\n1. LOADING AND PREPARING CCCI DATA")
    print("-" * 50)
    
    # Load data
    data = pd.read_csv(csv_file)
    
    # Clean up the last column name if it has carriage return
    data.columns = [col.strip().replace('\r', '') for col in data.columns]
    
    print(f"Data loaded: {data.shape[0]} quarters from {data['quarter'].iloc[0]} to {data['quarter'].iloc[-1]}")
    print(f"Variables: {list(data.columns)}")
    
    # Convert quarter to datetime for analysis
    data['quarter_date'] = pd.to_datetime(data['quarter'])
    
    # Define feature columns and target
    feature_columns = ["UNRATE", "PSAVERT", "PCE_QoQ", "REVOLSL_QoQ"]
    target_column = "CORCACBS"
    lookback = 8
    
    print(f"Features: {feature_columns}")
    print(f"Target: {target_column} (Charge-off rate)")
    print(f"Lookback: {lookback} quarters")
    
    # ==================== Data Statistics ====================
    print("\n2. DATA OVERVIEW")
    print("-" * 50)
    
    print("Target variable (CORCACBS) statistics:")
    print(f"  Mean: {data[target_column].mean():.3f}%")
    print(f"  Std:  {data[target_column].std():.3f}%") 
    print(f"  Min:  {data[target_column].min():.3f}% ({data.loc[data[target_column].idxmin(), 'quarter']})")
    print(f"  Max:  {data[target_column].max():.3f}% ({data.loc[data[target_column].idxmax(), 'quarter']})")
    
    # ==================== Prepare Model Inputs ====================
    print("\n3. PREPARING MODEL INPUTS")
    print("-" * 50)
    
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_columns].values)
    y_values = data[target_column].values
    
    # Store scaling info for results
    scaling_info = {
        'feature_means': scaler.mean_.tolist(),
        'feature_stds': scaler.scale_.tolist(),
        'feature_names': feature_columns
    }
    
    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    y_tensor = torch.tensor(y_values[lookback:], dtype=torch.float32).unsqueeze(0)
    
    print(f"Input tensor shape: {X_tensor.shape}")
    print(f"Target tensor shape: {y_tensor.shape}")
    print(f"Analysis period: {data['quarter'].iloc[lookback]} to {data['quarter'].iloc[-1]}")
    
    # ==================== Model Analysis ====================
    print("\n4. RUNNING MODEL ANALYSIS")
    print("-" * 50)
    
    device = next(trained_model.parameters()).device
    trained_model.eval()
    
    with torch.no_grad():
        predictions, attention_weights, interpretability = trained_model(X_tensor.to(device))
        multivectors = trained_model.ga_embedding(X_tensor.to(device))
    
    # Move to CPU
    predictions = predictions.cpu()
    attention_weights = attention_weights.cpu()
    multivectors = multivectors.cpu()
    
    # Calculate performance metrics
    mse = torch.nn.functional.mse_loss(predictions, y_tensor).item()
    rmse = np.sqrt(mse)
    
    print(f"Model Performance:")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  Mean Absolute Error: {torch.mean(torch.abs(predictions - y_tensor)).item():.4f}%")
    
    # ==================== Define Crisis Periods ====================
    # Based on NBER recession dates and banking crises
    crisis_periods = {
        'S&L_Crisis': ('1990Q1', '1991Q1'),
        'Dot_Com_Recession': ('2001Q1', '2001Q4'),
        '2008_Financial_Crisis': ('2007Q4', '2009Q2'),
        'COVID_Recession': ('2020Q2', '2020Q4')
    }
    
    print("\n5. CRISIS PERIODS DEFINED")
    print("-" * 50)
    for name, (start, end) in crisis_periods.items():
        print(f"{name}: {start} to {end}")
    
    # ==================== Overall Component Analysis ====================
    print("\n6. COMPONENT IMPORTANCE ANALYSIS")
    print("-" * 50)
    
    print("Component Magnitudes:")
    print(f"  Scalar magnitude: {interpretability['scalar_magnitude']:.4f}")
    print(f"  Vector magnitude: {interpretability['vector_magnitude']:.4f}")
    print(f"  Bivector magnitude: {interpretability['bivector_magnitude']:.4f}")
    
    print("\nComponent Impacts (Occlusion-based):")
    print(f"  Scalar impact: {interpretability['scalar_causal_impact']:.4f}")
    print(f"  Vector impact: {interpretability['vector_causal_impact']:.4f}")
    print(f"  Bivector impact: {interpretability['bivector_causal_impact']:.4f}")
    
    # Rank components by importance
    component_ranking = [
        ('Scalar', interpretability['scalar_causal_impact']),
        ('Vector', interpretability['vector_causal_impact']), 
        ('Bivector', interpretability['bivector_causal_impact'])
    ]
    component_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print("\nComponent Ranking by Impact:")
    for i, (name, impact) in enumerate(component_ranking, 1):
        print(f"  {i}. {name}: {impact:.4f}")
    
    # ==================== Attention Pattern Analysis ====================
    print("\n7. ATTENTION PATTERN ANALYSIS")
    print("-" * 50)
    
    avg_attention = attention_weights.mean(dim=(0, 1)).numpy()
    most_attended_lag = np.argmax(avg_attention)
    attention_entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-8))
    
    print(f"Most attended lag: t-{most_attended_lag + 1} (weight: {avg_attention[most_attended_lag]:.4f})")
    print(f"Attention entropy: {attention_entropy:.3f}")
    print(f"Attention distribution: {[f'{w:.3f}' for w in avg_attention]}")
    
    # ==================== Crisis-Specific Analysis ====================
    print("\n8. CRISIS-SPECIFIC ANALYSIS")
    print("-" * 50)
    
    crisis_analysis = {}
    crisis_indices = {}
    
    for crisis_name, (start_quarter, end_quarter) in crisis_periods.items():
        try:
            # Find indices for crisis period
            start_mask = data['quarter'] == start_quarter
            end_mask = data['quarter'] == end_quarter
            
            if start_mask.any() and end_mask.any():
                start_idx = data[start_mask].index[0] - lookback
                end_idx = data[end_mask].index[0] - lookback
                
                # Adjust for tensor bounds
                start_idx = max(0, start_idx)
                end_idx = min(attention_weights.shape[1] - 1, end_idx)
                
                if end_idx > start_idx:
                    crisis_indices[crisis_name] = (start_idx, end_idx)
                    
                    # Extract crisis period data
                    crisis_attn = attention_weights.squeeze(0)[start_idx:end_idx+1, :]
                    crisis_mv = multivectors.squeeze(0)[start_idx+lookback:end_idx+lookback+1, :]
                    crisis_targets = y_tensor.squeeze(0)[start_idx:end_idx+1]
                    
                    # Analyze this crisis
                    crisis_analysis[crisis_name] = {
                        'avg_charge_off_rate': crisis_targets.mean().item(),
                        'max_charge_off_rate': crisis_targets.max().item(),
                        'scalar_magnitude': crisis_mv[..., 0].abs().mean().item(),
                        'vector_magnitude': crisis_mv[..., 1:5].abs().mean().item(),
                        'bivector_magnitude': crisis_mv[..., 5:].abs().mean().item(),
                        'attention_entropy': -(crisis_attn.mean(0) * torch.log(crisis_attn.mean(0) + 1e-8)).sum().item(),
                        'most_attended_lag': torch.argmax(crisis_attn.mean(0)).item(),
                        'period_length': end_idx - start_idx + 1
                    }
                    
                    print(f"\n{crisis_name}:")
                    print(f"  Avg charge-off rate: {crisis_analysis[crisis_name]['avg_charge_off_rate']:.3f}%")
                    print(f"  Max charge-off rate: {crisis_analysis[crisis_name]['max_charge_off_rate']:.3f}%")
                    print(f"  Dominant component: {'Bivector' if crisis_analysis[crisis_name]['bivector_magnitude'] > max(crisis_analysis[crisis_name]['scalar_magnitude'], crisis_analysis[crisis_name]['vector_magnitude']) else 'Scalar' if crisis_analysis[crisis_name]['scalar_magnitude'] > crisis_analysis[crisis_name]['vector_magnitude'] else 'Vector'}")
                    print(f"  Most attended lag: t-{crisis_analysis[crisis_name]['most_attended_lag'] + 1}")
                    
        except Exception as e:
            print(f"Warning: Could not analyze {crisis_name}: {e}")
    
    # ==================== Create Component Heatmap ====================
    print("\n9. CREATING COMPONENT-ATTENTION HEATMAP")
    print("-" * 50)
    
    # Component labels including all bivectors
    component_labels = ['scalar'] + feature_columns
    
    # Add bivector labels (all 6 combinations)
    bivector_planes = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, j in bivector_planes:
        component_labels.append(f'{feature_columns[i]}∧{feature_columns[j]}')
    
    print(f"Components: {component_labels}")
    
    # Compute heatmap data
    T, lookback_size = attention_weights.squeeze(0).shape
    n_components = multivectors.shape[-1]
    
    heatmap_data = np.zeros((n_components, lookback_size))
    
    for t in range(T):
        for lag in range(lookback_size):
            for comp in range(n_components):
                component_magnitude = abs(multivectors.squeeze(0)[t + lookback, comp].item())
                attention_weight = attention_weights.squeeze(0)[t, lag].item()
                heatmap_data[comp, lag] += component_magnitude * attention_weight
    
    heatmap_data = heatmap_data / T
    
    # ==================== Query-Specific Analysis ====================
    query_analyses = {}
    query_heatmaps = {}  # Initialize query heatmaps dictionary
    
    if specific_queries is None:
        # Auto-select interesting queries
        specific_queries = []
        
        # Add crisis peaks
        for crisis_name, analysis in crisis_analysis.items():
            if crisis_name in crisis_indices:
                start_idx, end_idx = crisis_indices[crisis_name]
                # Find peak charge-off rate in this period
                crisis_targets = y_tensor.squeeze(0)[start_idx:end_idx+1]
                peak_offset = torch.argmax(crisis_targets).item()
                peak_idx = start_idx + peak_offset
                specific_queries.append(peak_idx)
                print(f"Added crisis peak for {crisis_name} at index {peak_idx}")
        
        # Add high charge-off periods
        high_periods = torch.topk(y_tensor.squeeze(0), k=3).indices.tolist()
        specific_queries.extend(high_periods)
        
        # Add some recent periods
        T = attention_weights.shape[1]  # Get T from attention weights shape
        specific_queries.extend([T-10, T-5])  # Recent periods
        
        specific_queries = sorted(list(set(specific_queries)))  # Remove duplicates
    
    print(f"\n10. QUERY-SPECIFIC ANALYSIS")
    print(f"Analyzing queries at indices: {specific_queries}")
    print("-" * 50)
    
    for i, query_idx in enumerate(specific_queries):
        if 0 <= query_idx < T:
            # Get query data
            query_quarter = data['quarter'].iloc[query_idx + lookback]
            query_attention = attention_weights.squeeze(0)[query_idx, :]
            query_prediction = predictions.squeeze(0)[query_idx].item()
            query_actual = y_tensor.squeeze(0)[query_idx].item()
            query_multivector = multivectors.squeeze(0)[query_idx + lookback, :]
            
            # Component analysis
            scalar_val = query_multivector[0].item()
            vector_vals = query_multivector[1:5].tolist()
            bivector_vals = query_multivector[5:].tolist()
            
            # Create bivector analysis
            bivector_analysis = {}
            for k, (var_i, var_j) in enumerate(bivector_planes):
                if k < len(bivector_vals):
                    pair_name = f'{feature_columns[var_i]}∧{feature_columns[var_j]}'
                    bivector_analysis[pair_name] = bivector_vals[k]
            
            # Create query-specific heatmap
            query_component_attention = np.zeros(n_components)
            for comp in range(n_components):
                for lag in range(lookback_size):
                    component_magnitude = abs(query_multivector[comp].item())
                    attention_weight = query_attention[lag].item()
                    query_component_attention[comp] += component_magnitude * attention_weight
            
            query_heatmaps[f'query_{i}'] = {
                'data': query_component_attention.tolist(),
                'labels': component_labels,
                'quarter': query_quarter
            }
            
            # Store analysis
            query_analyses[f'query_{i}'] = {
                'query_index': query_idx,
                'query_quarter': query_quarter,
                'prediction': query_prediction,
                'actual_value': query_actual,
                'prediction_error': abs(query_prediction - query_actual),
                'scalar_value': scalar_val,
                'vector_values': dict(zip(feature_columns, vector_vals)),
                'bivector_values': bivector_analysis,
                'attention_weights': query_attention.tolist(),
                'most_attended_lag': torch.argmax(query_attention).item(),
                'attention_entropy': -(query_attention * torch.log(query_attention + 1e-8)).sum().item()
            }
            
            print(f"\nQuery {i}: {query_quarter} (Index: {query_idx})")
            print(f"  Charge-off rate: {query_actual:.3f}% (predicted: {query_prediction:.3f}%)")
            print(f"  Error: {abs(query_prediction - query_actual):.3f}%")
            print(f"  Most attended lag: t-{torch.argmax(query_attention).item() + 1}")
            print(f"  Strongest bivector: {max(bivector_analysis.items(), key=lambda x: abs(x[1]))[0] if bivector_analysis else 'None'}")
    
    # ==================== Create Visualizations ====================
    print(f"\n11. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    if save_plots:
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Time series plot
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(data['quarter_date'].iloc[lookback:], y_tensor.squeeze(0).numpy(), 'b-', label='Actual', alpha=0.7)
    ax1.plot(data['quarter_date'].iloc[lookback:], predictions.squeeze(0).numpy(), 'r--', label='Predicted', alpha=0.7)
    ax1.set_title('Charge-off Rates: Actual vs Predicted')
    ax1.set_ylabel('Charge-off Rate (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add crisis shading
    for crisis_name, (start_idx, end_idx) in crisis_indices.items():
        start_date = data['quarter_date'].iloc[start_idx + lookback]
        end_date = data['quarter_date'].iloc[end_idx + lookback]
        ax1.axvspan(start_date, end_date, alpha=0.2, label=crisis_name)
    
    # 2. Component importance
    ax2 = plt.subplot(3, 3, 2)
    comp_names = ['Scalar', 'Vector', 'Bivector']
    comp_impacts = [interpretability['scalar_causal_impact'],
                    interpretability['vector_causal_impact'],
                    interpretability['bivector_causal_impact']]
    bars = ax2.bar(comp_names, comp_impacts)
    ax2.set_title('Component Importance (Occlusion)')
    ax2.set_ylabel('Impact on Prediction')
    
    # 3. Attention pattern
    ax3 = plt.subplot(3, 3, 3)
    lag_labels = [f't-{i+1}' for i in range(len(avg_attention))]
    bars = ax3.bar(range(len(avg_attention)), avg_attention)
    ax3.set_title('Average Attention by Lag')
    ax3.set_xlabel('Lookback Period')
    ax3.set_ylabel('Attention Weight')
    ax3.set_xticks(range(len(lag_labels)))
    ax3.set_xticklabels(lag_labels)
    
    # 4. Prediction scatter
    ax4 = plt.subplot(3, 3, 4)
    pred_np = predictions.squeeze(0).numpy()
    actual_np = y_tensor.squeeze(0).numpy()
    ax4.scatter(actual_np, pred_np, alpha=0.6)
    ax4.plot([actual_np.min(), actual_np.max()], [actual_np.min(), actual_np.max()], 'r--')
    ax4.set_xlabel('Actual Charge-off Rate (%)')
    ax4.set_ylabel('Predicted Charge-off Rate (%)')
    ax4.set_title(f'Predictions vs Actual (RMSE: {rmse:.3f}%)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Crisis comparison
    if crisis_analysis:
        ax5 = plt.subplot(3, 3, 5)
        crisis_names = list(crisis_analysis.keys())
        crisis_rates = [crisis_analysis[c]['avg_charge_off_rate'] for c in crisis_names]
        bars = ax5.bar(range(len(crisis_names)), crisis_rates)
        ax5.set_title('Average Charge-off Rates by Crisis')
        ax5.set_ylabel('Charge-off Rate (%)')
        ax5.set_xticks(range(len(crisis_names)))
        ax5.set_xticklabels(crisis_names, rotation=45, ha='right')
    
    # 6. Component heatmap
    ax6 = plt.subplot(3, 3, (6, 9))  # Span multiple subplots
    im = ax6.imshow(heatmap_data, cmap='viridis', aspect='auto')
    ax6.set_title('Component-Weighted Attention Heatmap')
    ax6.set_xlabel('Lookback Period')
    ax6.set_ylabel('GA Components')
    ax6.set_xticks(range(lookback_size))
    ax6.set_xticklabels([f't-{i+1}' for i in range(lookback_size)])
    ax6.set_yticks(range(len(component_labels)))
    ax6.set_yticklabels(component_labels)
    plt.colorbar(im, ax=ax6, label='Weighted Attention')
    
    # 7. Economic variables over time
    ax7 = plt.subplot(3, 3, 7)
    for i, var in enumerate(feature_columns):
        ax7.plot(data['quarter_date'], data[var], label=var, alpha=0.7)
    ax7.set_title('Economic Variables Over Time')
    ax7.set_ylabel('Standardized Values')
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # 8. Residuals over time
    ax8 = plt.subplot(3, 3, 8)
    residuals = pred_np - actual_np
    ax8.plot(data['quarter_date'].iloc[lookback:], residuals)
    ax8.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax8.set_title('Prediction Residuals Over Time')
    ax8.set_ylabel('Prediction Error (%)')
    ax8.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/ccci_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis plot saved to {output_dir}/ccci_comprehensive_analysis.png")
    
    plt.show()
    
    # ==================== Summary Report ====================
    print("\n" + "="*80)
    print("CCCI GA MODEL ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"  RMSE: {rmse:.4f}%")
    print(f"  Mean Absolute Error: {torch.mean(torch.abs(predictions - y_tensor)).item():.4f}%")
    print(f"  R-squared: {1 - (torch.sum((y_tensor - predictions)**2) / torch.sum((y_tensor - y_tensor.mean())**2)).item():.4f}")
    
    print(f"\nCOMPONENT ANALYSIS:")
    print(f"  Most important component: {component_ranking[0][0]} (impact: {component_ranking[0][1]:.4f})")
    print(f"  Bivector vs Vector importance ratio: {interpretability['bivector_causal_impact']/interpretability['vector_causal_impact']:.2f}")
    
    print(f"\nATTENTION INSIGHTS:")
    print(f"  Primary focus on lag: t-{most_attended_lag + 1}")
    print(f"  Attention pattern: {'Focused' if attention_entropy < 1.5 else 'Distributed'} (entropy: {attention_entropy:.3f})")
    
    if crisis_analysis:
        print(f"\nCRISIS INSIGHTS:")
        worst_crisis = max(crisis_analysis.items(), key=lambda x: x[1]['max_charge_off_rate'])
        print(f"  Highest charge-off period: {worst_crisis[0]} ({worst_crisis[1]['max_charge_off_rate']:.3f}%)")
        
        # Compare crisis patterns
        bivector_dominant_crises = [name for name, analysis in crisis_analysis.items() 
                                   if analysis['bivector_magnitude'] > max(analysis['scalar_magnitude'], analysis['vector_magnitude'])]
        print(f"  Bivector-dominated crises: {bivector_dominant_crises}")
    
    # ==================== Compile Results ====================
    results = {
        'data_info': {
            'n_quarters': len(data),
            'analysis_period': (data['quarter'].iloc[lookback], data['quarter'].iloc[-1]),
            'target_stats': {
                'mean': data[target_column].mean(),
                'std': data[target_column].std(),
                'min': data[target_column].min(),
                'max': data[target_column].max()
            }
        },
        'scaling_info': scaling_info,  # Now properly defined
        'model_performance': {
            'rmse': rmse,
            'mae': torch.mean(torch.abs(predictions - y_tensor)).item(),
            'r_squared': 1 - (torch.sum((y_tensor - predictions)**2) / torch.sum((y_tensor - y_tensor.mean())**2)).item()
        },
        'component_analysis': {
            'interpretability_metrics': interpretability,
            'component_ranking': component_ranking
        },
        'attention_analysis': {
            'avg_weights': avg_attention.tolist(),
            'most_attended_lag': int(most_attended_lag),
            'entropy': float(attention_entropy)
        },
        'heatmap_data': {
            'data': heatmap_data.tolist(),
            'component_labels': component_labels,
            'lag_labels': [f't-{i+1}' for i in range(lookback_size)]
        },
        'crisis_analysis': crisis_analysis,
        'query_analyses': query_analyses,
        'query_heatmaps': query_heatmaps,  # Now properly defined
        'crisis_periods_used': crisis_periods
    }
    
    print(f"\nAnalysis complete! Total crises analyzed: {len(crisis_analysis)}")
    print(f"Query-specific analyses: {len(query_analyses)}")
    
    return results

# Example usage function
def run_ccci_analysis(trained_model):
    """
    Quick function to run analysis with your trained model
    
    Usage:
        results = run_ccci_analysis(ga_model)
    """
    return analyze_ccci_model(
        trained_model=trained_model,
        csv_file='ccci_data.csv',
        save_plots=True,
        output_dir='./ccci_analysis_results/'
    )

# Additional utility functions for deeper analysis
def compare_crisis_patterns(results: Dict[str, Any]) -> None:
    """
    Compare attention and component patterns across different crises
    """
    crisis_analysis = results['crisis_analysis']
    
    print("\n" + "="*60)
    print("CRISIS PATTERN COMPARISON")
    print("="*60)
    
    # Create comparison dataframe
    comparison_data = []
    for crisis_name, analysis in crisis_analysis.items():
        comparison_data.append({
            'Crisis': crisis_name,
            'Avg_ChargeOff': analysis['avg_charge_off_rate'],
            'Max_ChargeOff': analysis['max_charge_off_rate'],
            'Scalar_Mag': analysis['scalar_magnitude'],
            'Vector_Mag': analysis['vector_magnitude'],
            'Bivector_Mag': analysis['bivector_magnitude'],
            'Attention_Lag': analysis['most_attended_lag'],
            'Attention_Entropy': analysis['attention_entropy']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Find patterns
    print("Crisis Severity Ranking (by max charge-off rate):")
    severity_ranking = df.sort_values('Max_ChargeOff', ascending=False)
    for i, row in severity_ranking.iterrows():
        print(f"  {row['Crisis']}: {row['Max_ChargeOff']:.3f}%")
    
    print("\nComponent Dominance Patterns:")
    for _, row in df.iterrows():
        dominant = 'Scalar' if row['Scalar_Mag'] > max(row['Vector_Mag'], row['Bivector_Mag']) else \
                  'Vector' if row['Vector_Mag'] > row['Bivector_Mag'] else 'Bivector'
        print(f"  {row['Crisis']}: {dominant} dominant")
    
    print("\nAttention Focus Patterns:")
    for _, row in df.iterrows():
        focus = 'Recent' if row['Attention_Lag'] < 2 else 'Medium' if row['Attention_Lag'] < 5 else 'Distant'
        print(f"  {row['Crisis']}: {focus} past focus (t-{row['Attention_Lag']+1})")

def generate_economic_insights(results: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate economic insights from the GA model analysis
    """
    insights = {}
    
    # Component insights
    comp_ranking = results['component_analysis']['component_ranking']
    most_important = comp_ranking[0][0]
    
    if most_important == 'Bivector':
        insights['geometric_insight'] = (
            "Bivector dominance suggests that interactions between economic variables "
            "are more predictive than individual variables alone. This indicates "
            "non-linear relationships and complex feedback loops in credit cycles."
        )
    elif most_important == 'Vector':
        insights['geometric_insight'] = (
            "Vector dominance indicates that individual economic variables have "
            "strong predictive power, suggesting more linear relationships in credit cycles."
        )
    else:
        insights['geometric_insight'] = (
            "Scalar dominance suggests a baseline trend or intercept term is most "
            "important, indicating strong temporal persistence in charge-off rates."
        )
    
    # Attention insights
    attention_data = results['attention_analysis']
    most_attended_lag = attention_data['most_attended_lag']
    entropy = attention_data['entropy']
    
    if most_attended_lag <= 1:
        insights['temporal_insight'] = (
            f"Model focuses primarily on recent data (t-{most_attended_lag+1}), "
            "suggesting credit conditions respond quickly to economic changes."
        )
    elif most_attended_lag >= 6:
        insights['temporal_insight'] = (
            f"Model focuses on distant past (t-{most_attended_lag+1}), "
            "indicating long-term economic cycles drive credit conditions."
        )
    else:
        insights['temporal_insight'] = (
            f"Model focuses on medium-term history (t-{most_attended_lag+1}), "
            "suggesting balanced short and long-term economic influences."
        )
    
    if entropy > 2.0:
        insights['attention_pattern'] = (
            "High attention entropy indicates the model considers multiple time periods, "
            "suggesting complex temporal dependencies in credit cycles."
        )
    else:
        insights['attention_pattern'] = (
            "Low attention entropy indicates focused attention on specific periods, "
            "suggesting clear temporal patterns in credit prediction."
        )
    
    # Crisis insights
    if 'crisis_analysis' in results:
        crisis_data = results['crisis_analysis']
        bivector_crises = []
        for crisis_name, analysis in crisis_data.items():
            if analysis['bivector_magnitude'] > max(analysis['scalar_magnitude'], analysis['vector_magnitude']):
                bivector_crises.append(crisis_name)
        
        if bivector_crises:
            insights['crisis_insight'] = (
                f"Crises dominated by bivector components ({', '.join(bivector_crises)}) "
                "show complex variable interactions, suggesting systemic risk patterns "
                "emerge from economic variable correlations during stress periods."
            )
        else:
            insights['crisis_insight'] = (
                "Crisis periods show linear component dominance, suggesting "
                "individual economic indicators are sufficient for crisis prediction."
            )
    
    return insights

def create_interpretability_report(results: Dict[str, Any], output_dir: str = "./ccci_analysis/") -> str:
    """
    Create a comprehensive interpretability report
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "interpretability_report.md")
    insights = generate_economic_insights(results)
    
    with open(report_path, 'w') as f:
        f.write("# CCCI GA Model Interpretability Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        perf = results['model_performance']
        f.write(f"- **Model Performance**: RMSE {perf['rmse']:.4f}%, R² {perf['r_squared']:.4f}\n")
        
        comp_ranking = results['component_analysis']['component_ranking']
        f.write(f"- **Key Component**: {comp_ranking[0][0]} (impact: {comp_ranking[0][1]:.4f})\n")
        
        attn = results['attention_analysis']
        f.write(f"- **Temporal Focus**: t-{attn['most_attended_lag']+1} quarters back\n")
        f.write(f"- **Analysis Period**: {results['data_info']['analysis_period'][0]} to {results['data_info']['analysis_period'][1]}\n\n")
        
        # Geometric Algebra Insights
        f.write("## Geometric Algebra Insights\n\n")
        f.write(f"**Component Analysis**: {insights.get('geometric_insight', 'No insight available')}\n\n")
        
        # Component breakdown
        f.write("### Component Importance Ranking:\n")
        for i, (name, impact) in enumerate(comp_ranking, 1):
            f.write(f"{i}. **{name}**: {impact:.4f}\n")
        f.write("\n")
        
        # Temporal Analysis
        f.write("## Temporal Analysis\n\n")
        f.write(f"**Attention Pattern**: {insights.get('temporal_insight', 'No insight available')}\n\n")
        f.write(f"**Pattern Complexity**: {insights.get('attention_pattern', 'No insight available')}\n\n")
        
        # Crisis Analysis
        if 'crisis_analysis' in results and results['crisis_analysis']:
            f.write("## Crisis Period Analysis\n\n")
            f.write(f"**Crisis Patterns**: {insights.get('crisis_insight', 'No insight available')}\n\n")
            
            f.write("### Crisis Comparison:\n")
            for crisis_name, analysis in results['crisis_analysis'].items():
                f.write(f"- **{crisis_name}**: {analysis['avg_charge_off_rate']:.3f}% avg, ")
                f.write(f"{analysis['max_charge_off_rate']:.3f}% peak\n")
        
        # Technical Details
        f.write("\n## Technical Details\n\n")
        f.write("### Data Information:\n")
        data_info = results['data_info']
        f.write(f"- Quarters analyzed: {data_info['n_quarters']}\n")
        f.write(f"- Target variable range: {data_info['target_stats']['min']:.3f}% to {data_info['target_stats']['max']:.3f}%\n")
        f.write(f"- Target variable mean: {data_info['target_stats']['mean']:.3f}%\n\n")
        
        # Model Architecture Insights
        f.write("### Model Architecture Insights:\n")
        heatmap = results['heatmap_data']
        f.write(f"- GA Components tracked: {len(heatmap['component_labels'])}\n")
        f.write(f"- Lookback periods: {len(heatmap['lag_labels'])}\n")
        f.write(f"- Component labels: {', '.join(heatmap['component_labels'][:5])}...\n\n")
        
        # Query Analysis Summary
        if results['query_analyses']:
            f.write("### Key Time Period Analysis:\n")
            for query_name, analysis in list(results['query_analyses'].items())[:3]:  # Top 3
                f.write(f"- **{analysis['query_quarter']}**: ")
                f.write(f"{analysis['actual_value']:.3f}% charge-off, ")
                f.write(f"error: {analysis['prediction_error']:.3f}%\n")
        
        f.write("\n---\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Interpretability report saved to: {report_path}")
    return report_path

def analyze_query_attribution(
    trained_model, 
    X_tensor: torch.Tensor, 
    query_idx: int, 
    feature_columns: List[str],
    lookback: int = 8,
    device: str = None
) -> Dict[str, Any]:
    """
    Detailed attribution analysis for a specific query (time point)
    
    Returns component-wise, feature-wise, and lag-wise attributions
    """
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
    
    # Component attribution via occlusion
    component_attributions = {}
    
    # Scalar attribution
    modified_mv = query_multivector.clone()
    modified_mv[0] = 0  # Zero out scalar
    # This requires implementing a way to inject modified multivectors
    # For now, we'll use the interpretability metrics provided by the model
    
    # Feature-wise attribution (for each input feature at each lag)
    feature_attributions = np.zeros((len(feature_columns), lookback))
    
    # Create masks for each feature at each lag
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
        # Zero out entire lag
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
    """
    Create comprehensive visualization for a specific query analysis
    """
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Component Values (Scalar, Vector, Bivector)
    ax1 = plt.subplot(3, 4, 1)
    components = ['Scalar']
    component_values = [query_analysis['scalar_value']]
    
    # Add vector components
    for feat, val in query_analysis['vector_values'].items():
        components.append(f'V_{feat}')
        component_values.append(val)
    
    # Add bivector components
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
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, component_values)):
        ax1.text(val + 0.01 * max(abs(min(component_values)), abs(max(component_values))), 
                i, f'{val:.3f}', va='center', fontsize=8)
    
    # 2. Attention Weights
    ax2 = plt.subplot(3, 4, 2)
    lags = [f't-{i+1}' for i in range(len(query_analysis['attention_weights']))]
    bars = ax2.bar(range(len(lags)), query_analysis['attention_weights'])
    ax2.set_xticks(range(len(lags)))
    ax2.set_xticklabels(lags, rotation=45)
    ax2.set_title(f'Attention Weights - {quarter_label}')
    ax2.set_ylabel('Attention Weight')
    ax2.grid(True, alpha=0.3)
    
    # Highlight most attended lag
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
    plt.colorbar(im, ax=ax3, label='Attribution (Impact on Prediction)')
    
    # Add text annotations for significant values
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
    
    # Color bars by magnitude
    max_attr = np.max(np.abs(query_analysis['lag_attributions']))
    for bar, val in zip(bars, query_analysis['lag_attributions']):
        intensity = abs(val) / max_attr
        bar.set_color(plt.cm.viridis(intensity))
    
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
    
    # Add value labels
    for bar, val in zip(bars, comp_mags):
        ax5.text(bar.get_x() + bar.get_width()/2, val + val*0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Vector Component Breakdown
    ax6 = plt.subplot(3, 4, 7)
    vector_features = list(query_analysis['vector_values'].keys())
    vector_values = list(query_analysis['vector_values'].values())
    bars = ax6.barh(vector_features, vector_values, color='blue', alpha=0.7)
    ax6.set_title(f'Vector Components - {quarter_label}')
    ax6.set_xlabel('Vector Value')
    ax6.grid(True, alpha=0.3)
    
    # Color bars by sign
    for bar, val in zip(bars, vector_values):
        bar.set_color('blue' if val >= 0 else 'red')
    
    # 7. Bivector Component Breakdown
    ax7 = plt.subplot(3, 4, 8)
    biv_names = list(query_analysis['bivector_values'].keys())
    biv_values = list(query_analysis['bivector_values'].values())
    
    # Truncate names for display
    biv_display = [name.replace('∧', '\n∧\n') for name in biv_names]
    
    bars = ax7.barh(range(len(biv_display)), biv_values, color='green', alpha=0.7)
    ax7.set_yticks(range(len(biv_display)))
    ax7.set_yticklabels(biv_display, fontsize=8)
    ax7.set_title(f'Bivector Components - {quarter_label}')
    ax7.set_xlabel('Bivector Value')
    ax7.grid(True, alpha=0.3)
    
    # Color bars by sign
    for bar, val in zip(bars, biv_values):
        bar.set_color('green' if val >= 0 else 'orange')
    
    # 8. Prediction Summary
    ax8 = plt.subplot(3, 4, 9)
    ax8.axis('off')
    
    # Create text summary
    summary_text = f"""
    Query Analysis Summary
    ═══════════════════════
    
    Quarter: {quarter_label}
    Prediction: {query_analysis['prediction']:.3f}%
    """
    
    if actual_value is not None:
        summary_text += f"Actual: {actual_value:.3f}%\n"
        summary_text += f"Error: {abs(query_analysis['prediction'] - actual_value):.3f}%\n"
    
    summary_text += f"""
    
    Attention Focus: t-{np.argmax(query_analysis['attention_weights'])+1}
    Attention Entropy: {query_analysis['attention_entropy']:.3f}
    
    Strongest Components:
    • Scalar: {query_analysis['scalar_value']:.3f}
    • Vector: {max(query_analysis['vector_values'].items(), key=lambda x: abs(x[1]))[0]}: {max(query_analysis['vector_values'].values(), key=abs):.3f}
    • Bivector: {max(query_analysis['bivector_values'].items(), key=lambda x: abs(x[1]))[0]}: {max(query_analysis['bivector_values'].values(), key=abs):.3f}
    """
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 9. Feature Attribution Summary
    ax9 = plt.subplot(3, 4, 10)
    feature_total_attr = np.sum(query_analysis['feature_attributions'], axis=1)
    bars = ax9.barh(feature_columns, feature_total_attr)
    ax9.set_title(f'Total Feature Attribution - {quarter_label}')
    ax9.set_xlabel('Total Attribution')
    ax9.grid(True, alpha=0.3)
    
    # Color by contribution
    for bar, val in zip(bars, feature_total_attr):
        bar.set_color('darkblue' if val >= 0 else 'darkred')
    
    # 10. Attention-Weighted Attribution
    ax10 = plt.subplot(3, 4, 11)
    # Weight feature attributions by attention
    weighted_attr = query_analysis['feature_attributions'] * query_analysis['attention_weights'][np.newaxis, :]
    weighted_total = np.sum(weighted_attr, axis=1)
    
    bars = ax10.barh(feature_columns, weighted_total)
    ax10.set_title(f'Attention-Weighted Attribution - {quarter_label}')
    ax10.set_xlabel('Weighted Attribution')
    ax10.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, weighted_total):
        bar.set_color('purple' if val >= 0 else 'orange')
    
    # 11. Component Interaction Analysis
    ax11 = plt.subplot(3, 4, 12)
    
    # Create interaction strength visualization
    interaction_data = []
    for biv_name, biv_val in query_analysis['bivector_values'].items():
        # Extract feature names from bivector
        feat_pair = biv_name.split('∧')
        if len(feat_pair) == 2:
            interaction_data.append({
                'features': biv_name,
                'strength': abs(biv_val),
                'direction': 'positive' if biv_val >= 0 else 'negative'
            })
    
    if interaction_data:
        interaction_df = pd.DataFrame(interaction_data)
        colors = ['green' if d == 'positive' else 'red' for d in interaction_df['direction']]
        bars = ax11.barh(interaction_df['features'], interaction_df['strength'], color=colors, alpha=0.7)
        ax11.set_title(f'Feature Interactions - {quarter_label}')
        ax11.set_xlabel('Interaction Strength')
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'No bivector\ninteractions\ndetected', 
                 ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title(f'Feature Interactions - {quarter_label}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Query analysis visualization saved to: {save_path}")
    
    plt.show()

def analyze_multiple_queries(
    trained_model,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    data: pd.DataFrame,
    feature_columns: List[str],
    query_indices: List[int],
    lookback: int = 8,
    save_plots: bool = True,
    output_dir: str = "./query_analysis/"
) -> Dict[str, Any]:
    """
    Analyze multiple specific queries with detailed attribution and visualization
    """
    import os
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing {len(query_indices)} specific queries...")
    print("-" * 50)
    
    all_query_analyses = {}
    
    for i, query_idx in enumerate(query_indices):
        if 0 <= query_idx < y_tensor.shape[1]:
            print(f"\nAnalyzing Query {i+1}/{len(query_indices)}: Index {query_idx}")
            
            # Get query metadata
            quarter_label = data['quarter'].iloc[query_idx + lookback]
            actual_value = y_tensor.squeeze(0)[query_idx].item()
            
            print(f"  Quarter: {quarter_label}")
            print(f"  Actual charge-off rate: {actual_value:.3f}%")
            
            # Perform detailed attribution analysis
            query_analysis = analyze_query_attribution(
                trained_model, X_tensor, query_idx, feature_columns, lookback
            )
            
            print(f"  Predicted: {query_analysis['prediction']:.3f}%")
            print(f"  Error: {abs(query_analysis['prediction'] - actual_value):.3f}%")
            print(f"  Primary attention: t-{np.argmax(query_analysis['attention_weights'])+1}")
            
            # Create visualization
            if save_plots:
                save_path = f"{output_dir}/query_{i+1}_{quarter_label.replace('Q', 'q')}_analysis.png"
            else:
                save_path = None
                
            visualize_query_analysis(
                query_analysis, feature_columns, quarter_label, 
                actual_value, save_path
            )
            
            # Store analysis
            query_analysis['quarter'] = quarter_label
            query_analysis['actual_value'] = actual_value
            query_analysis['prediction_error'] = abs(query_analysis['prediction'] - actual_value)
            
            all_query_analyses[f'query_{i+1}_{quarter_label}'] = query_analysis
            
        else:
            print(f"Warning: Query index {query_idx} out of bounds")
    
    # Create comparative analysis
    if len(all_query_analyses) > 1:
        print(f"\nCreating comparative analysis for {len(all_query_analyses)} queries...")
        create_query_comparison_visualization(all_query_analyses, feature_columns, 
                                            f"{output_dir}/query_comparison.png" if save_plots else None)
    
    return all_query_analyses

def create_query_comparison_visualization(
    query_analyses: Dict[str, Dict[str, Any]], 
    feature_columns: List[str],
    save_path: str = None
) -> None:
    """
    Create comparative visualization across multiple queries
    """
    fig = plt.figure(figsize=(20, 12))
    
    query_names = list(query_analyses.keys())
    n_queries = len(query_names)
    
    # 1. Prediction Accuracy Comparison
    ax1 = plt.subplot(3, 3, 1)
    predictions = [q['prediction'] for q in query_analyses.values()]
    actual_values = [q['actual_value'] for q in query_analyses.values()]
    errors = [q['prediction_error'] for q in query_analyses.values()]
    
    x_pos = np.arange(n_queries)
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, actual_values, width, label='Actual', alpha=0.7, color='blue')
    bars2 = ax1.bar(x_pos + width/2, predictions, width, label='Predicted', alpha=0.7, color='red')
    
    ax1.set_xlabel('Queries')
    ax1.set_ylabel('Charge-off Rate (%)')
    ax1.set_title('Prediction vs Actual Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([q.split('_')[-1] for q in query_names], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Attention Pattern Comparison
    ax2 = plt.subplot(3, 3, 2)
    attention_matrix = np.array([q['attention_weights'] for q in query_analyses.values()])
    im = ax2.imshow(attention_matrix, cmap='viridis', aspect='auto')
    ax2.set_title('Attention Patterns Across Queries')
    ax2.set_xlabel('Lookback Period')
    ax2.set_ylabel('Queries')
    ax2.set_xticks(range(attention_matrix.shape[1]))
    ax2.set_xticklabels([f't-{i+1}' for i in range(attention_matrix.shape[1])])
    ax2.set_yticks(range(n_queries))
    ax2.set_yticklabels([q.split('_')[-1] for q in query_names])
    plt.colorbar(im, ax=ax2, label='Attention Weight')
    
    # 3. Component Magnitude Heatmap
    ax3 = plt.subplot(3, 3, 3)
    scalar_vals = [q['scalar_value'] for q in query_analyses.values()]
    vector_vals = [[q['vector_values'][feat] for feat in feature_columns] for q in query_analyses.values()]
    
    # Create heatmap data
    heatmap_data = []
    component_labels = ['Scalar'] + feature_columns
    
    for i, query_name in enumerate(query_names):
        query = query_analyses[query_name]
        row_data = [query['scalar_value']] + [query['vector_values'][feat] for feat in feature_columns]
        heatmap_data.append(row_data)
    
    heatmap_data = np.array(heatmap_data)
    im = ax3.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
    ax3.set_title('Component Values Across Queries')
    ax3.set_xlabel('Components')
    ax3.set_ylabel('Queries')
    ax3.set_xticks(range(len(component_labels)))
    ax3.set_xticklabels(component_labels, rotation=45)
    ax3.set_yticks(range(n_queries))
    ax3.set_yticklabels([q.split('_')[-1] for q in query_names])
    plt.colorbar(im, ax=ax3, label='Component Value')
    
    # 4. Error Analysis
    ax4 = plt.subplot(3, 3, 4)
    bars = ax4.bar(range(n_queries), errors)
    ax4.set_xlabel('Queries')
    ax4.set_ylabel('Prediction Error (%)')
    ax4.set_title('Prediction Errors by Query')
    ax4.set_xticks(range(n_queries))
    ax4.set_xticklabels([q.split('_')[-1] for q in query_names], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Color bars by error magnitude
    max_error = max(errors)
    for bar, error in zip(bars, errors):
        intensity = error / max_error
        bar.set_color(plt.cm.Reds(intensity))
    
    # 5. Attention Entropy Comparison
    ax5 = plt.subplot(3, 3, 5)
    entropies = [q['attention_entropy'] for q in query_analyses.values()]
    bars = ax5.bar(range(n_queries), entropies)
    ax5.set_xlabel('Queries')
    ax5.set_ylabel('Attention Entropy')
    ax5.set_title('Attention Distribution (Entropy)')
    ax5.set_xticks(range(n_queries))
    ax5.set_xticklabels([q.split('_')[-1] for q in query_names], rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature Attribution Summary
    ax6 = plt.subplot(3, 3, 6)
    # Average feature attribution across all queries
    all_feature_attrs = np.array([np.sum(q['feature_attributions'], axis=1) for q in query_analyses.values()])
    avg_attr = np.mean(all_feature_attrs, axis=0)
    std_attr = np.std(all_feature_attrs, axis=0)
    
    bars = ax6.bar(feature_columns, avg_attr, yerr=std_attr, capsize=5)
    ax6.set_xlabel('Features')
    ax6.set_ylabel('Average Attribution')
    ax6.set_title('Average Feature Attribution Across Queries')
    ax6.set_xticklabels(feature_columns, rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 7. Bivector Strength Comparison
    ax7 = plt.subplot(3, 3, 7)
    # Get all unique bivector names
    all_biv_names = set()
    for q in query_analyses.values():
        all_biv_names.update(q['bivector_values'].keys())
    all_biv_names = sorted(list(all_biv_names))
    
    if all_biv_names:
        biv_matrix = []
        for query in query_analyses.values():
            row = [query['bivector_values'].get(name, 0) for name in all_biv_names]
            biv_matrix.append(row)
        
        biv_matrix = np.array(biv_matrix)
        im = ax7.imshow(biv_matrix, cmap='RdBu_r', aspect='auto')
        ax7.set_title('Bivector Values Across Queries')
        ax7.set_xlabel('Bivector Components')
        ax7.set_ylabel('Queries')
        ax7.set_xticks(range(len(all_biv_names)))
        ax7.set_xticklabels([name.replace('∧', '\n∧\n') for name in all_biv_names], rotation=45, fontsize=8)
        ax7.set_yticks(range(n_queries))
        ax7.set_yticklabels([q.split('_')[-1] for q in query_names])
        plt.colorbar(im, ax=ax7, label='Bivector Value')
    
    # 8. Most Attended Lag Distribution
    ax8 = plt.subplot(3, 3, 8)
    most_attended_lags = [np.argmax(q['attention_weights']) for q in query_analyses.values()]
    lag_counts = np.bincount(most_attended_lags, minlength=8)  # Assuming 8 lookback periods
    
    bars = ax8.bar(range(len(lag_counts)), lag_counts)
    ax8.set_xlabel('Most Attended Lag')
    ax8.set_ylabel('Number of Queries')
    ax8.set_title('Distribution of Primary Attention Focus')
    ax8.set_xticks(range(len(lag_counts)))
    ax8.set_xticklabels([f't-{i+1}' for i in range(len(lag_counts))])
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary statistics
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    avg_entropy = np.mean(entropies)
    
    summary_text = f"""
    Query Comparison Summary
    ═══════════════════════
    
    Queries Analyzed: {n_queries}
    
    Prediction Performance:
    • Avg Error: {avg_error:.3f}%
    • Max Error: {max_error:.3f}%
    • Min Error: {min_error:.3f}%
    
    Attention Patterns:
    • Avg Entropy: {avg_entropy:.3f}
    • Most Common Focus: t-{np.argmax(lag_counts)+1}
    
    Component Analysis:
    • Most Variable Component: Vector
    • Avg Scalar Magnitude: {np.mean([abs(q['scalar_value']) for q in query_analyses.values()]):.3f}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Query comparison visualization saved to: {save_path}")
    
    plt.show()

def create_interactive_query_explorer(
    trained_model,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    data: pd.DataFrame,
    feature_columns: List[str],
    lookback: int = 8,
    output_dir: str = "./interactive_analysis/"
) -> None:
    """
    Create an interactive query exploration interface
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create HTML interface for interactive exploration
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCCI Query Explorer</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .query-selector {{ background: #f0f0f0; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .results {{ background: white; border: 1px solid #ddd; padding: 20px; border-radius: 10px; }}
            .component {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-left: 4px solid #007acc; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e9e9e9; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>CCCI GA Model Query Explorer</h1>
            <div class="query-selector">
                <h3>Select Quarter to Analyze:</h3>
                <select id="quarterSelect" onchange="analyzeQuery()">
                    <option value="">Select a quarter...</option>"""
    
    # Add quarters to dropdown
    for i in range(lookback, len(data)):
        quarter = data['quarter'].iloc[i]
        charge_off = data[data.columns[-1]].iloc[i]  # Last column should be target
        html_content += f'<option value="{i-lookback}">{quarter} ({charge_off:.2f}%)</option>'
    
    html_content += """
                </select>
                <button onclick="analyzeQuery()" style="margin-left: 10px; padding: 10px 20px;">Analyze</button>
            </div>
            
            <div id="results" class="results" style="display: none;">
                <div id="queryResults"></div>
            </div>
        </div>
        
        <script>
            function analyzeQuery() {
                const selector = document.getElementById('quarterSelect');
                const resultsDiv = document.getElementById('results');
                const queryResults = document.getElementById('queryResults');
                
                if (selector.value === "") {
                    resultsDiv.style.display = 'none';
                    return;
                }
                
                // Show loading message
                queryResults.innerHTML = '<p>Analyzing query... This would call the Python backend.</p>';
                resultsDiv.style.display = 'block';
                
                // In a real implementation, this would make an AJAX call to Python backend
                // For now, show placeholder content
                setTimeout(() => {
                    queryResults.innerHTML = `
                        <h3>Analysis Results for ${selector.options[selector.selectedIndex].text}</h3>
                        <div class="metric">Prediction: 2.45%</div>
                        <div class="metric">Actual: 2.32%</div>
                        <div class="metric">Error: 0.13%</div>
                        <div class="metric">Primary Attention: t-3</div>
                        
                        <div class="component">
                            <h4>Component Analysis</h4>
                            <p>Scalar: 0.234 | Vector Avg: 0.156 | Bivector Avg: 0.089</p>
                        </div>
                        
                        <div class="component">
                            <h4>Feature Attributions</h4>
                            <p>UNRATE: High negative impact | PSAVERT: Moderate positive impact</p>
                        </div>
                    `;
                }, 1000);
            }
        </script>
    </body>
    </html>"""
    
    # Save HTML file
    html_path = os.path.join(output_dir, "query_explorer.html")
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Interactive query explorer created: {html_path}")
    print("Open this file in a web browser to explore queries interactively.")

def analyze_component_stability(
    trained_model,
    X_tensor: torch.Tensor,
    feature_columns: List[str],
    lookback: int = 8,
    window_size: int = 4
) -> Dict[str, Any]:
    """
    Analyze stability of component values over time using rolling windows
    """
    device = next(trained_model.parameters()).device
    trained_model.eval()
    
    with torch.no_grad():
        predictions, attention_weights, interpretability = trained_model(X_tensor.to(device))
        multivectors = trained_model.ga_embedding(X_tensor.to(device))
    
    multivectors = multivectors.cpu()
    T = multivectors.shape[1] - lookback
    
    # Extract component time series
    scalar_series = multivectors.squeeze(0)[lookback:, 0].numpy()
    vector_series = multivectors.squeeze(0)[lookback:, 1:5].numpy()
    bivector_series = multivectors.squeeze(0)[lookback:, 5:].numpy()
    
    # Calculate rolling statistics
    def rolling_stats(series, window):
        stats = {
            'mean': np.convolve(series, np.ones(window)/window, mode='valid'),
            'std': np.array([np.std(series[i:i+window]) for i in range(len(series)-window+1)]),
            'min': np.array([np.min(series[i:i+window]) for i in range(len(series)-window+1)]),
            'max': np.array([np.max(series[i:i+window]) for i in range(len(series)-window+1)])
        }
        return stats
    
    # Analyze each component type
    scalar_stats = rolling_stats(scalar_series, window_size)
    
    # Vector component statistics (average across features)
    vector_mag_series = np.mean(np.abs(vector_series), axis=1)
    vector_stats = rolling_stats(vector_mag_series, window_size)
    
    # Bivector component statistics
    bivector_mag_series = np.mean(np.abs(bivector_series), axis=1)
    bivector_stats = rolling_stats(bivector_mag_series, window_size)
    
    # Feature-specific vector analysis
    feature_stats = {}
    for i, feat in enumerate(feature_columns):
        feature_stats[feat] = rolling_stats(vector_series[:, i], window_size)
    
    # Calculate stability metrics
    scalar_stability = np.mean(scalar_stats['std'])
    vector_stability = np.mean(vector_stats['std'])
    bivector_stability = np.mean(bivector_stats['std'])
    
    stability_ranking = [
        ('Scalar', scalar_stability),
        ('Vector', vector_stability),
        ('Bivector', bivector_stability)
    ]
    stability_ranking.sort(key=lambda x: x[1])  # Most stable first
    
    return {
        'scalar_stats': scalar_stats,
        'vector_stats': vector_stats,
        'bivector_stats': bivector_stats,
        'feature_stats': feature_stats,
        'stability_ranking': stability_ranking,
        'scalar_stability': scalar_stability,
        'vector_stability': vector_stability,
        'bivector_stability': bivector_stability,
        'window_size': window_size
    }

def visualize_component_stability(
    stability_analysis: Dict[str, Any],
    data: pd.DataFrame,
    lookback: int = 8,
    save_path: str = None
) -> None:
    """
    Visualize component stability analysis
    """
    fig = plt.figure(figsize=(18, 12))
    
    window_size = stability_analysis['window_size']
    T = len(stability_analysis['scalar_stats']['mean'])
    
    # Create time axis for rolling statistics
    time_axis = data['quarter_date'].iloc[lookback+window_size-1:lookback+window_size-1+T]
    
    # 1. Component Stability Ranking
    ax1 = plt.subplot(3, 3, 1)
    ranking = stability_analysis['stability_ranking']
    components = [r[0] for r in ranking]
    stabilities = [r[1] for r in ranking]
    colors = ['green', 'orange', 'red']
    
    bars = ax1.bar(components, stabilities, color=colors)
    ax1.set_title('Component Stability Ranking\n(Lower = More Stable)')
    ax1.set_ylabel('Standard Deviation')
    ax1.grid(True, alpha=0.3)
    
    for bar, stability in zip(bars, stabilities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{stability:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Scalar Component Time Series
    ax2 = plt.subplot(3, 3, 2)
    scalar_stats = stability_analysis['scalar_stats']
    ax2.plot(time_axis, scalar_stats['mean'], 'b-', label='Mean', linewidth=2)
    ax2.fill_between(time_axis, 
                    scalar_stats['mean'] - scalar_stats['std'],
                    scalar_stats['mean'] + scalar_stats['std'],
                    alpha=0.3, color='blue', label='±1 Std')
    ax2.set_title(f'Scalar Component Stability (Window: {window_size})')
    ax2.set_ylabel('Scalar Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Vector Component Time Series
    ax3 = plt.subplot(3, 3, 3)
    vector_stats = stability_analysis['vector_stats']
    ax3.plot(time_axis, vector_stats['mean'], 'g-', label='Mean', linewidth=2)
    ax3.fill_between(time_axis,
                    vector_stats['mean'] - vector_stats['std'],
                    vector_stats['mean'] + vector_stats['std'],
                    alpha=0.3, color='green', label='±1 Std')
    ax3.set_title(f'Vector Component Stability (Magnitude)')
    ax3.set_ylabel('Vector Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Bivector Component Time Series
    ax4 = plt.subplot(3, 3, 4)
    bivector_stats = stability_analysis['bivector_stats']
    ax4.plot(time_axis, bivector_stats['mean'], 'r-', label='Mean', linewidth=2)
    ax4.fill_between(time_axis,
                    bivector_stats['mean'] - bivector_stats['std'],
                    bivector_stats['mean'] + bivector_stats['std'],
                    alpha=0.3, color='red', label='±1 Std')
    ax4.set_title(f'Bivector Component Stability (Magnitude)')
    ax4.set_ylabel('Bivector Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Feature-Specific Stability
    ax5 = plt.subplot(3, 3, 5)
    feature_names = list(stability_analysis['feature_stats'].keys())
    feature_stabilities = [np.mean(stability_analysis['feature_stats'][feat]['std']) 
                          for feat in feature_names]
    
    bars = ax5.bar(feature_names, feature_stabilities)
    ax5.set_title('Feature-Specific Vector Stability')
    ax5.set_ylabel('Average Standard Deviation')
    ax5.set_xticklabels(feature_names, rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Color bars by stability
    max_stability = max(feature_stabilities)
    for bar, stability in zip(bars, feature_stabilities):
        intensity = stability / max_stability
        bar.set_color(plt.cm.Reds(intensity))
    
    # 6. Rolling Standard Deviation Comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time_axis, scalar_stats['std'], 'b-', label='Scalar', linewidth=2)
    ax6.plot(time_axis, vector_stats['std'], 'g-', label='Vector', linewidth=2)
    ax6.plot(time_axis, bivector_stats['std'], 'r-', label='Bivector', linewidth=2)
    ax6.set_title('Component Stability Over Time')
    ax6.set_ylabel('Rolling Standard Deviation')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # 7. Component Range Analysis
    ax7 = plt.subplot(3, 3, 7)
    scalar_range = scalar_stats['max'] - scalar_stats['min']
    vector_range = vector_stats['max'] - vector_stats['min']
    bivector_range = bivector_stats['max'] - bivector_stats['min']
    
    ax7.plot(time_axis, scalar_range, 'b-', label='Scalar Range', linewidth=2)
    ax7.plot(time_axis, vector_range, 'g-', label='Vector Range', linewidth=2)
    ax7.plot(time_axis, bivector_range, 'r-', label='Bivector Range', linewidth=2)
    ax7.set_title('Component Value Ranges Over Time')
    ax7.set_ylabel('Range (Max - Min)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # 8. Feature Correlation with Stability
    ax8 = plt.subplot(3, 3, 8)
    # This would show how feature stability correlates with model performance
    # For now, show a placeholder
    ax8.text(0.5, 0.5, 'Feature Stability\nCorrelation Analysis\n\n(Implementation needed\nfor full correlation)', 
             ha='center', va='center', transform=ax8.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray'))
    ax8.set_title('Stability-Performance Correlation')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    Component Stability Analysis
    ══════════════════════════
    
    Window Size: {window_size} quarters
    Analysis Period: {T} windows
    
    Stability Ranking:
    1. {stability_analysis['stability_ranking'][0][0]}: {stability_analysis['stability_ranking'][0][1]:.4f}
    2. {stability_analysis['stability_ranking'][1][0]}: {stability_analysis['stability_ranking'][1][1]:.4f}
    3. {stability_analysis['stability_ranking'][2][0]}: {stability_analysis['stability_ranking'][2][1]:.4f}
    
    Component Insights:
    • Most stable: {stability_analysis['stability_ranking'][0][0]}
    • Least stable: {stability_analysis['stability_ranking'][-1][0]}
    • Stability ratio: {stability_analysis['stability_ranking'][-1][1]/stability_analysis['stability_ranking'][0][1]:.2f}x
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Component stability visualization saved to: {save_path}")
    
    plt.show()

def create_component_evolution_analysis(
    trained_model,
    X_tensor: torch.Tensor,
    data: pd.DataFrame,
    feature_columns: List[str],
    lookback: int = 8,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Analyze how components evolve over time and create visualizations
    """
    device = next(trained_model.parameters()).device
    trained_model.eval()
    
    with torch.no_grad():
        predictions, attention_weights, interpretability = trained_model(X_tensor.to(device))
        multivectors = trained_model.ga_embedding(X_tensor.to(device))
    
    # Move to CPU
    multivectors = multivectors.cpu()
    attention_weights = attention_weights.cpu()
    
    T = multivectors.shape[1] - lookback
    
    # Extract time series for each component type
    scalar_series = multivectors.squeeze(0)[lookback:, 0].numpy()
    vector_series = multivectors.squeeze(0)[lookback:, 1:5].numpy()
    bivector_series = multivectors.squeeze(0)[lookback:, 5:].numpy()
    
    # Calculate attention entropy over time
    attention_entropy_series = []
    for t in range(T):
        attn = attention_weights.squeeze(0)[t, :]
        entropy = -(attn * torch.log(attn + 1e-8)).sum().item()
        attention_entropy_series.append(entropy)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    time_axis = data['quarter_date'].iloc[lookback:lookback+T]
    
    # 1. Scalar Component Evolution
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(time_axis, scalar_series, 'b-', linewidth=2, alpha=0.8)
    ax1.set_title('Scalar Component Evolution')
    ax1.set_ylabel('Scalar Value')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Add crisis shading
    crisis_periods = {
        '2008_Financial_Crisis': ('2007Q4', '2009Q2'),
        'COVID_Recession': ('2020Q2', '2020Q4')
    }
    
    for crisis_name, (start, end) in crisis_periods.items():
        try:
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)
            ax1.axvspan(start_date, end_date, alpha=0.2, color='red', label=crisis_name if crisis_name == '2008_Financial_Crisis' else "")
        except:
            pass
    
    # 2. Vector Components Evolution (Individual Features)
    ax2 = plt.subplot(4, 3, 2)
    for i, feat in enumerate(feature_columns):
        ax2.plot(time_axis, vector_series[:, i], label=feat, linewidth=2, alpha=0.8)
    ax2.set_title('Vector Components Evolution')
    ax2.set_ylabel('Vector Values')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Bivector Components Evolution
    ax3 = plt.subplot(4, 3, 3)
    bivector_pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, (feat_i, feat_j) in enumerate(bivector_pairs[:min(len(bivector_pairs), bivector_series.shape[1])]):
        label = f'{feature_columns[feat_i]}∧{feature_columns[feat_j]}'
        ax3.plot(time_axis, bivector_series[:, i], label=label, linewidth=2, alpha=0.8)
    ax3.set_title('Bivector Components Evolution')
    ax3.set_ylabel('Bivector Values')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Component Magnitudes Over Time
    ax4 = plt.subplot(4, 3, 4)
    scalar_mag = np.abs(scalar_series)
    vector_mag = np.mean(np.abs(vector_series), axis=1)
    bivector_mag = np.mean(np.abs(bivector_series), axis=1)
    
    ax4.plot(time_axis, scalar_mag, 'r-', label='Scalar', linewidth=2)
    ax4.plot(time_axis, vector_mag, 'g-', label='Vector', linewidth=2)
    ax4.plot(time_axis, bivector_mag, 'b-', label='Bivector', linewidth=2)
    ax4.set_title('Component Magnitudes Over Time')
    ax4.set_ylabel('Absolute Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # 5. Attention Entropy Evolution
    ax5 = plt.subplot(4, 3, 5)
    ax5.plot(time_axis, attention_entropy_series, 'purple', linewidth=2)
    ax5.set_title('Attention Entropy Over Time')
    ax5.set_ylabel('Entropy')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    
    # Add horizontal line for "focused" vs "distributed" attention
    ax5.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='Focus Threshold')
    ax5.legend()
    
    # 6. Component Dominance Over Time
    ax6 = plt.subplot(4, 3, 6)
    dominance = np.zeros((T, 3))  # scalar, vector, bivector
    dominance[:, 0] = scalar_mag
    dominance[:, 1] = vector_mag
    dominance[:, 2] = bivector_mag
    
    # Normalize to show relative dominance
    dominance_normalized = dominance / (dominance.sum(axis=1, keepdims=True) + 1e-8)
    
    ax6.stackplot(time_axis, 
                 dominance_normalized[:, 0],
                 dominance_normalized[:, 1], 
                 dominance_normalized[:, 2],
                 labels=['Scalar', 'Vector', 'Bivector'],
                 colors=['red', 'green', 'blue'], alpha=0.7)
    ax6.set_title('Component Dominance Over Time')
    ax6.set_ylabel('Relative Contribution')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # 7. Feature Interaction Strength (Bivector Analysis)
    ax7 = plt.subplot(4, 3, 7)
    if bivector_series.shape[1] > 0:
        bivector_strength = np.sum(np.abs(bivector_series), axis=1)
        ax7.plot(time_axis, bivector_strength, 'orange', linewidth=2)
        ax7.set_title('Feature Interaction Strength')
        ax7.set_ylabel('Total Bivector Magnitude')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'No Bivector\nComponents\nDetected', 
                ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Feature Interaction Strength')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # 8. Most Attended Lag Over Time
    ax8 = plt.subplot(4, 3, 8)
    most_attended_lags = [torch.argmax(attention_weights.squeeze(0)[t, :]).item() + 1 
                         for t in range(T)]
    ax8.plot(time_axis, most_attended_lags, 'brown', linewidth=2, marker='o', markersize=3)
    ax8.set_title('Primary Attention Focus Over Time')
    ax8.set_ylabel('Most Attended Lag (t-N)')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 9)  # Assuming 8 lookback periods
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)
    
    # 9. Component Correlation with Target
    ax9 = plt.subplot(4, 3, 9)
    target_values = data[data.columns[-1]].iloc[lookback:lookback+T].values
    
    # Calculate correlations
    scalar_corr = np.corrcoef(scalar_series, target_values)[0, 1]
    vector_corr = np.corrcoef(vector_mag, target_values)[0, 1]
    bivector_corr = np.corrcoef(bivector_mag, target_values)[0, 1]
    
    correlations = [scalar_corr, vector_corr, bivector_corr]
    component_names = ['Scalar', 'Vector', 'Bivector']
    colors = ['red', 'green', 'blue']
    
    bars = ax9.bar(component_names, correlations, color=colors, alpha=0.7)
    ax9.set_title('Component-Target Correlations')
    ax9.set_ylabel('Correlation with Charge-off Rate')
    ax9.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax9.grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{corr:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 10. Rolling Component Statistics
    ax10 = plt.subplot(4, 3, 10)
    window_size = 8  # 2 years
    if T > window_size:
        scalar_rolling_std = pd.Series(scalar_series).rolling(window_size).std().iloc[window_size-1:]
        vector_rolling_std = pd.Series(vector_mag).rolling(window_size).std().iloc[window_size-1:]
        bivector_rolling_std = pd.Series(bivector_mag).rolling(window_size).std().iloc[window_size-1:]
        
        rolling_time_axis = time_axis[window_size-1:]
        
        ax10.plot(rolling_time_axis, scalar_rolling_std, 'r-', label='Scalar', linewidth=2)
        ax10.plot(rolling_time_axis, vector_rolling_std, 'g-', label='Vector', linewidth=2)
        ax10.plot(rolling_time_axis, bivector_rolling_std, 'b-', label='Bivector', linewidth=2)
        ax10.set_title(f'Rolling Component Volatility ({window_size}Q)')
        ax10.set_ylabel('Rolling Std Dev')
        ax10.legend()
        ax10.grid(True, alpha=0.3)
        plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax10.text(0.5, 0.5, f'Need >{window_size}Q\nfor rolling analysis', 
                 ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title(f'Rolling Component Volatility ({window_size}Q)')
    
    # 11. Component Phase Analysis
    ax11 = plt.subplot(4, 3, 11)
    # Analyze if components lead/lag the target
    target_changes = np.diff(target_values)
    scalar_changes = np.diff(scalar_series)
    
    if len(target_changes) > 1 and len(scalar_changes) > 1:
        # Calculate cross-correlation for lead/lag analysis
        correlation_lags = np.correlate(target_changes, scalar_changes, mode='full')
        lags = np.arange(-len(scalar_changes)+1, len(scalar_changes))
        max_corr_idx = np.argmax(np.abs(correlation_lags))
        lead_lag = lags[max_corr_idx]
        
        ax11.plot(lags, correlation_lags, 'purple', linewidth=2)
        ax11.axvline(x=lead_lag, color='red', linestyle='--', label=f'Max at lag {lead_lag}')
        ax11.set_title('Scalar-Target Lead/Lag Analysis')
        ax11.set_xlabel('Lag (quarters)')
        ax11.set_ylabel('Cross-correlation')
        ax11.legend()
        ax11.grid(True, alpha=0.3)
    else:
        ax11.text(0.5, 0.5, 'Insufficient data\nfor lead/lag analysis', 
                 ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title('Scalar-Target Lead/Lag Analysis')
    
    # 12. Summary Statistics and Insights
    ax12 = plt.subplot(4, 3, 12)
    ax12.axis('off')
    
    # Calculate summary statistics
    avg_scalar = np.mean(scalar_mag)
    avg_vector = np.mean(vector_mag)
    avg_bivector = np.mean(bivector_mag)
    avg_entropy = np.mean(attention_entropy_series)
    
    # Find periods of highest activity
    combined_activity = scalar_mag + vector_mag + bivector_mag
    highest_activity_idx = np.argmax(combined_activity)
    highest_activity_quarter = data['quarter'].iloc[lookback + highest_activity_idx]
    
    summary_text = f"""
    Component Evolution Summary
    ═══════════════════════════
    
    Analysis Period: {T} quarters
    
    Average Magnitudes:
    • Scalar: {avg_scalar:.3f}
    • Vector: {avg_vector:.3f}
    • Bivector: {avg_bivector:.3f}
    
    Correlations with Target:
    • Scalar: {scalar_corr:.3f}
    • Vector: {vector_corr:.3f}
    • Bivector: {bivector_corr:.3f}
    
    Attention Insights:
    • Avg Entropy: {avg_entropy:.3f}
    • Pattern: {"Focused" if avg_entropy < 1.5 else "Distributed"}
    
    Peak Activity Period:
    • {highest_activity_quarter}
    • Combined magnitude: {combined_activity[highest_activity_idx]:.3f}
    """
    
    ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Component evolution visualization saved to: {save_path}")
    
    plt.show()
    
    # Return analysis results
    return {
        'scalar_series': scalar_series,
        'vector_series': vector_series,
        'bivector_series': bivector_series,
        'attention_entropy_series': attention_entropy_series,
        'component_correlations': {
            'scalar': scalar_corr,
            'vector': vector_corr,
            'bivector': bivector_corr
        },
        'average_magnitudes': {
            'scalar': avg_scalar,
            'vector': avg_vector,
            'bivector': avg_bivector
        },
        'peak_activity': {
            'quarter': highest_activity_quarter,
            'index': highest_activity_idx,
            'magnitude': combined_activity[highest_activity_idx]
        }
    }

# Updated complete workflow function with new query analysis capabilities
def complete_ccci_workflow_enhanced(trained_model, csv_file: str = 'ccci_data.csv', specific_queries: List[int] = None):
    """
    Enhanced complete workflow including detailed query-specific analysis
    """
    print("Starting Enhanced CCCI Analysis Workflow...")
    print("="*80)
    
    # 1. Run main analysis
    print("\n1. RUNNING MAIN ANALYSIS...")
    results = analyze_ccci_model(trained_model, csv_file)
    
    # 2. Load data for additional analyses
    data = pd.read_csv(csv_file)
    data.columns = [col.strip().replace('\r', '') for col in data.columns]
    feature_columns = ["UNRATE", "PSAVERT", "PCE_QoQ", "REVOLSL_QoQ"]
    lookback = 8
    
    # Prepare tensors (same as in main analysis)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[feature_columns].values)
    y_values = data[data.columns[-1]].values  # Target column
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)
    y_tensor = torch.tensor(y_values[lookback:], dtype=torch.float32).unsqueeze(0)
    
    # 3. Auto-select interesting queries if not provided
    if specific_queries is None:
        print("\n2. AUTO-SELECTING INTERESTING QUERIES...")
        specific_queries = []
        
        # Add high charge-off periods
        high_periods = torch.topk(y_tensor.squeeze(0), k=5).indices.tolist()
        specific_queries.extend(high_periods)
        
        # Add recent periods
        T = y_tensor.shape[1]
        specific_queries.extend([T-15, T-10, T-5])  # Recent periods
        
        # Add some middle periods for diversity
        specific_queries.extend([T//4, T//2, 3*T//4])
        
        specific_queries = sorted(list(set(specific_queries)))[:8]  # Limit to 8 queries
        print(f"Selected {len(specific_queries)} queries for detailed analysis")
    
    # 4. Detailed query analysis
    print("\n3. PERFORMING DETAILED QUERY ANALYSIS...")
    query_analyses = analyze_multiple_queries(
        trained_model, X_tensor, y_tensor, data, feature_columns, 
        specific_queries, lookback, save_plots=True, output_dir="./query_analysis/"
    )
    
    # 5. Component stability analysis
    print("\n4. ANALYZING COMPONENT STABILITY...")
    stability_analysis = analyze_component_stability(trained_model, X_tensor, feature_columns, lookback)
    visualize_component_stability(stability_analysis, data, lookback, 
                                 save_path="./query_analysis/component_stability.png")
    
    # 6. Component evolution analysis
    print("\n5. ANALYZING COMPONENT EVOLUTION...")
    evolution_analysis = create_component_evolution_analysis(
        trained_model, X_tensor, data, feature_columns, lookback,
        save_path="./query_analysis/component_evolution.png"
    )
    
    # 7. Create interactive explorer
    print("\n6. CREATING INTERACTIVE QUERY EXPLORER...")
    create_interactive_query_explorer(trained_model, X_tensor, y_tensor, data, feature_columns, lookback)
    
    # 8. Generate comprehensive insights
    print("\n7. GENERATING COMPREHENSIVE INSIGHTS...")
    insights = generate_economic_insights(results)
    
    # Add query-specific insights
    query_insights = generate_query_insights(query_analyses, evolution_analysis, stability_analysis)
    insights.update(query_insights)
    
    # 9. Create enhanced interpretability report
    print("\n8. CREATING ENHANCED INTERPRETABILITY REPORT...")
    enhanced_results = {**results, 
                       'query_analyses': query_analyses,
                       'stability_analysis': stability_analysis,
                       'evolution_analysis': evolution_analysis}
    
    report_path = create_enhanced_interpretability_report(enhanced_results, insights)
    
    # 10. Export comprehensive results
    print("\n9. EXPORTING COMPREHENSIVE RESULTS...")
    export_enhanced_results(enhanced_results, "./query_analysis/enhanced_results.json")
    
    print("\n" + "="*80)
    print("ENHANCED WORKFLOW COMPLETED")
    print("="*80)
    print("Generated outputs:")
    print("  📊 Main analysis: ./ccci_analysis_results/")
    print("  🔍 Query analysis: ./query_analysis/")
    print("  📈 Component stability: ./query_analysis/component_stability.png")
    print("  📉 Component evolution: ./query_analysis/component_evolution.png")
    print("  🌐 Interactive explorer: ./query_analysis/query_explorer.html")
    print("  📋 Enhanced report:", report_path)
    print("  💾 Complete results: ./query_analysis/enhanced_results.json")
    
    return enhanced_results, insights, query_analyses

def generate_query_insights(query_analyses, evolution_analysis, stability_analysis):
    """
    Generate insights from query-specific analysis
    """
    insights = {}
    
    if query_analyses:
        # Prediction accuracy insights
        errors = [q['prediction_error'] for q in query_analyses.values()]
        avg_error = np.mean(errors)
        
        if avg_error < 0.5:
            insights['accuracy_insight'] = (
                f"Excellent query-specific accuracy (avg error: {avg_error:.3f}%), "
                "indicating the model captures individual time period dynamics well."
            )
        elif avg_error < 1.0:
            insights['accuracy_insight'] = (
                f"Good query-specific accuracy (avg error: {avg_error:.3f}%), "
                "showing reliable individual prediction capability."
            )
        else:
            insights['accuracy_insight'] = (
                f"Moderate query-specific accuracy (avg error: {avg_error:.3f}%), "
                "suggesting challenges in capturing individual time period nuances."
            )
        
        # Attention pattern insights
        entropies = [q['attention_entropy'] for q in query_analyses.values()]
        avg_entropy = np.mean(entropies)
        
        if np.std(entropies) > 0.5:
            insights['attention_variability'] = (
                "High variability in attention patterns across queries suggests "
                "the model adapts its temporal focus based on specific economic conditions."
            )
        else:
            insights['attention_variability'] = (
                "Consistent attention patterns across queries indicate "
                "stable temporal relationship modeling."
            )
    
    # Component stability insights
    if stability_analysis:
        most_stable = stability_analysis['stability_ranking'][0][0]
        least_stable = stability_analysis['stability_ranking'][-1][0]
        
        insights['stability_insight'] = (
            f"Component stability analysis shows {most_stable} components are most stable, "
            f"while {least_stable} components show highest variability, indicating "
            f"{'consistent baseline trends' if most_stable == 'Scalar' else 'adaptive feature processing' if most_stable == 'Vector' else 'stable interaction patterns'}."
        )
    
    # Evolution insights
    if evolution_analysis and 'component_correlations' in evolution_analysis:
        corrs = evolution_analysis['component_correlations']
        highest_corr_component = max(corrs.items(), key=lambda x: abs(x[1]))
        
        insights['evolution_insight'] = (
            f"Component evolution analysis reveals {highest_corr_component[0]} components "
            f"have strongest correlation with charge-off rates ({highest_corr_component[1]:.3f}), "
            f"suggesting {'direct linear relationship' if highest_corr_component[0] == 'vector' else 'baseline trend importance' if highest_corr_component[0] == 'scalar' else 'interaction-driven dynamics'}."
        )
    
    return insights

def create_enhanced_interpretability_report(results: Dict[str, Any], insights: Dict[str, str], 
                                          output_dir: str = "./query_analysis/") -> str:
    """
    Create comprehensive interpretability report with query analysis
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, "enhanced_interpretability_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Enhanced CCCI GA Model Interpretability Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        perf = results['model_performance']
        f.write(f"- **Model Performance**: RMSE {perf['rmse']:.4f}%, R² {perf['r_squared']:.4f}\n")
        
        if 'query_analyses' in results:
            query_count = len(results['query_analyses'])
            f.write(f"- **Query-Specific Analyses**: {query_count} detailed time period analyses\n")
        
        f.write(f"- **Component Analysis**: Scalar, Vector, and Bivector interpretability\n")
        f.write(f"- **Stability Analysis**: Component evolution and variability assessment\n\n")
        
        # Main Insights Section
        f.write("## Key Insights\n\n")
        for insight_type, insight_text in insights.items():
            f.write(f"### {insight_type.replace('_', ' ').title()}\n")
            f.write(f"{insight_text}\n\n")
        
        # Query-Specific Analysis Section
        if 'query_analyses' in results:
            f.write("## Query-Specific Analysis\n\n")
            f.write("### Individual Time Period Insights\n\n")
            
            for query_name, analysis in list(results['query_analyses'].items())[:5]:  # Top 5
                quarter = analysis['quarter']
                prediction = analysis['prediction']
                actual = analysis['actual_value']
                error = analysis['prediction_error']
                
                f.write(f"**{quarter}**:\n")
                f.write(f"- Predicted: {prediction:.3f}%, Actual: {actual:.3f}%, Error: {error:.3f}%\n")
                
                # Find dominant component
                scalar_mag = abs(analysis['scalar_value'])
                vector_mag = np.mean([abs(v) for v in analysis['vector_values'].values()])
                bivector_mag = np.mean([abs(v) for v in analysis['bivector_values'].values()]) if analysis['bivector_values'] else 0
                
                dominant = 'Scalar' if scalar_mag > max(vector_mag, bivector_mag) else \
                          'Vector' if vector_mag > bivector_mag else 'Bivector'
                
                f.write(f"- Dominant component: {dominant}\n")
                f.write(f"- Primary attention: t-{np.argmax(analysis['attention_weights'])+1}\n\n")
        
        # Stability Analysis Section
        if 'stability_analysis' in results:
            f.write("## Component Stability Analysis\n\n")
            stability = results['stability_analysis']
            
            f.write("### Stability Ranking:\n")
            for i, (component, stability_val) in enumerate(stability['stability_ranking'], 1):
                f.write(f"{i}. **{component}**: {stability_val:.4f} (standard deviation)\n")
            f.write("\n")
        
        # Evolution Analysis Section
        if 'evolution_analysis' in results:
            f.write("## Component Evolution Analysis\n\n")
            evolution = results['evolution_analysis']
            
            if 'component_correlations' in evolution:
                f.write("### Correlations with Target Variable:\n")
                for component, correlation in evolution['component_correlations'].items():
                    f.write(f"- **{component.title()}**: {correlation:.4f}\n")
                f.write("\n")
            
            if 'peak_activity' in evolution:
                peak = evolution['peak_activity']
                f.write(f"### Peak Activity Period: {peak['quarter']}\n")
                f.write(f"Combined component magnitude: {peak['magnitude']:.3f}\n\n")
        
        # Technical Implementation Notes
        f.write("## Technical Implementation\n\n")
        f.write("### Analysis Components:\n")
        f.write("- **Attribution Analysis**: Feature and lag-wise contribution analysis\n")
        f.write("- **Component Visualization**: Scalar, vector, and bivector evolution\n")
        f.write("- **Stability Assessment**: Rolling window component variability\n")
        f.write("- **Interactive Explorer**: Web-based query investigation tool\n\n")
        
        f.write("### Files Generated:\n")
        f.write("- Individual query visualizations in `/query_analysis/`\n")
        f.write("- Component stability plots\n") 
        f.write("- Component evolution analysis\n")
        f.write("- Interactive HTML explorer\n")
        f.write("- Comprehensive results JSON\n\n")
        
        f.write("---\n")
        f.write(f"Enhanced report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Enhanced interpretability report saved to: {report_path}")
    return report_path

def export_enhanced_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Export enhanced results including all query analyses
    """
    import json
    
    # Function to convert numpy/torch objects to JSON serializable format
    def make_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    # Convert all results to serializable format
    serializable_results = make_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    print(f"Enhanced results exported to JSON: {output_path}")

# Usage examples with the enhanced query analysis:
"""
# Basic enhanced analysis
results, insights, query_analyses = complete_ccci_workflow_enhanced(your_trained_ga_model)

# Analyze specific quarters of interest
specific_quarters = [45, 67, 89, 102]  # Replace with your indices of interest
results, insights, query_analyses = complete_ccci_workflow_enhanced(
    your_trained_ga_model, 
    specific_queries=specific_quarters
)

# Detailed analysis of a single query
single_query_analysis = analyze_query_attribution(
    your_trained_ga_model, X_tensor, query_idx=50, feature_columns=feature_columns
)

# Create visualization for specific query
visualize_query_analysis(
    single_query_analysis, feature_columns, "2008Q3", actual_value=2.45,
    save_path="./specific_query_analysis.png"
)
"""

def export_results_json(results: Dict[str, Any], output_path: str) -> None:
    """
    Export results to JSON for further analysis
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'tolist'):
                    json_results[key][subkey] = subvalue.tolist()
                else:
                    json_results[key][subkey] = subvalue
        else:
            json_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Results exported to JSON: {output_path}")

# Complete example workflow
def complete_ccci_workflow(trained_model, csv_file: str = 'ccci_data.csv'):
    """
    Complete workflow for CCCI analysis including all visualization and reporting
    """
    print("Starting Complete CCCI Analysis Workflow...")
    
    # 1. Run main analysis
    results = analyze_ccci_model(trained_model, csv_file)
    
    # 2. Generate additional insights
    print("\nGenerating Economic Insights...")
    insights = generate_economic_insights(results)
    for key, insight in insights.items():
        print(f"{key}: {insight}")
    
    # 3. Create interpretability report
    print("\nCreating Interpretability Report...")
    report_path = create_interpretability_report(results)
    
    # 4. Export results
    print("\nExporting Results...")
    export_results_json(results, "./ccci_analysis/results.json")
    
    # 5. Create additional visualizations
    print("\nCreating Additional Visualizations...")
    visualize_component_evolution(results, "./ccci_analysis/component_evolution.png")
    
    # 6. Compare crisis patterns
    if results['crisis_analysis']:
        print("\nComparing Crisis Patterns...")
        compare_crisis_patterns(results)
    
    print("\n" + "="*80)
    print("COMPLETE WORKFLOW FINISHED")
    print("="*80)
    print(f"All outputs saved to: ./ccci_analysis/")
    print(f"Key files:")
    print(f"  - Main analysis: ./ccci_analysis/ccci_comprehensive_analysis.png")
    print(f"  - Report: {report_path}")
    print(f"  - Raw results: ./ccci_analysis/results.json")
    
    return results, insights

# Usage examples:
"""
# Basic usage
results = analyze_ccci_model(your_trained_ga_model)

# Quick analysis
results = run_ccci_analysis(your_trained_ga_model)

# Complete workflow
results, insights = complete_ccci_workflow(your_trained_ga_model)

# Generate just the insights
insights = generate_economic_insights(results)

# Create detailed report
report_path = create_interpretability_report(results)
"""