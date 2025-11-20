#!/usr/bin/env python3
"""
Analysis Script for Exercise 2
Evaluates LLM classification performance using multiple metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

def calculate_specificity(y_true, y_pred, average='macro'):
    """
    Calculate specificity (true negative rate) for multi-class classification

    Specificity = TN / (TN + FP)
    For multi-class, we calculate per-class and average
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    specificities = []
    for cls in classes:
        # Create binary classification for this class
        y_true_binary = (y_true == cls).astype(int)
        y_pred_binary = (y_pred == cls).astype(int)

        # Calculate confusion matrix components
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))

        # Calculate specificity for this class
        if (tn + fp) > 0:
            spec = tn / (tn + fp)
        else:
            spec = 0.0

        specificities.append(spec)

    if average == 'macro':
        return np.mean(specificities)
    elif average == 'weighted':
        # Weight by class frequency
        class_counts = [np.sum(y_true == cls) for cls in classes]
        return np.average(specificities, weights=class_counts)
    else:
        return specificities

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate a single model/prompt combination

    Returns dictionary with all metrics
    """
    # Remove None/NaN predictions (failed classifications)
    valid_idx = [i for i, pred in enumerate(y_pred) if pred is not None and not (isinstance(pred, float) and np.isnan(pred))]
    y_true_valid = y_true[valid_idx]
    y_pred_valid = np.array([y_pred[i] for i in valid_idx])

    # Calculate metrics
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    f1_macro = f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
    sensitivity = recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0)
    specificity = calculate_specificity(y_true_valid, y_pred_valid, average='macro')

    # Count failed predictions
    failed_preds = len(y_pred) - len(valid_idx)

    return {
        'model': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'valid_predictions': len(valid_idx),
        'failed_predictions': failed_preds,
        'success_rate': len(valid_idx) / len(y_pred)
    }

def analyze_results(predictions_file='data/leg_predictions.feather',
                     token_usage_file='data/token_usage.csv'):
    """
    Analyze all model/prompt combinations
    """
    print("Loading predictions...")
    df = pd.read_feather(predictions_file)

    # Load token usage if available
    token_df = None
    if Path(token_usage_file).exists():
        token_df = pd.read_csv(token_usage_file)
        print("Loaded token usage data")

    # Get true labels
    y_true = df['policy'].values

    # Get all prediction columns (exclude metadata columns)
    pred_cols = [col for col in df.columns
                 if col not in ['id', 'description', 'policy', 'policy_label']]

    # Evaluate each model/prompt combination
    results = []
    for col in pred_cols:
        y_pred = df[col].values
        metrics = evaluate_model(y_true, y_pred, col)
        results.append(metrics)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Sort by accuracy
    results_df = results_df.sort_values('accuracy', ascending=False)

    # Add token usage and cost estimates if available
    if token_df is not None:
        # Merge token usage
        for idx, row in results_df.iterrows():
            model_name = row['model']
            if model_name in token_df.columns:
                tokens = token_df[model_name].values[0]
                results_df.loc[idx, 'total_tokens'] = tokens

                # Estimate costs (approximate pricing as of 2024)
                if 'gpt4o_mini' in model_name:
                    # GPT-4o-mini: $0.15/1M input, $0.60/1M output
                    # Assume roughly 80% input, 20% output
                    cost = (tokens * 0.8 * 0.15 / 1_000_000) + (tokens * 0.2 * 0.60 / 1_000_000)
                elif 'gpt4o' in model_name:
                    # GPT-4o: $2.50/1M input, $10.00/1M output
                    cost = (tokens * 0.8 * 2.50 / 1_000_000) + (tokens * 0.2 * 10.00 / 1_000_000)
                elif 'gpt4_turbo' in model_name:
                    # GPT-4-turbo: $10.00/1M input, $30.00/1M output
                    cost = (tokens * 0.8 * 10.00 / 1_000_000) + (tokens * 0.2 * 30.00 / 1_000_000)
                else:
                    cost = 0

                results_df.loc[idx, 'estimated_cost_usd'] = cost

        # Calculate value metrics
        if 'estimated_cost_usd' in results_df.columns:
            results_df['accuracy_per_dollar'] = results_df['accuracy'] / results_df['estimated_cost_usd']
            results_df['f1_per_dollar'] = results_df['f1_macro'] / results_df['estimated_cost_usd']

    return results_df, df

def create_visualizations(results_df, output_dir='figures'):
    """
    Create visualizations comparing model performance
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Parse model and prompt from model name
    results_df['base_model'] = results_df['model'].str.extract(r'(gpt\d+[a-z_]*)')
    results_df['prompt_type'] = results_df['model'].str.extract(r'_(simple|detailed|reasoning)$')

    # 1. Performance comparison heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Accuracy heatmap
    pivot_acc = results_df.pivot(index='prompt_type', columns='base_model', values='accuracy')
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 0], vmin=0, vmax=1)
    axes[0, 0].set_title('Accuracy by Model and Prompt Type')

    # F1-Score heatmap
    pivot_f1 = results_df.pivot(index='prompt_type', columns='base_model', values='f1_macro')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[0, 1], vmin=0, vmax=1)
    axes[0, 1].set_title('F1-Score (Macro) by Model and Prompt Type')

    # Sensitivity heatmap
    pivot_sens = results_df.pivot(index='prompt_type', columns='base_model', values='sensitivity')
    sns.heatmap(pivot_sens, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 0], vmin=0, vmax=1)
    axes[1, 0].set_title('Sensitivity (Recall) by Model and Prompt Type')

    # Specificity heatmap
    pivot_spec = results_df.pivot(index='prompt_type', columns='base_model', values='specificity')
    sns.heatmap(pivot_spec, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1, 1], vmin=0, vmax=1)
    axes[1, 1].set_title('Specificity by Model and Prompt Type')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"Saved performance heatmaps to {output_dir}/performance_heatmaps.png")

    # 2. Bar plot comparing all metrics
    fig, ax = plt.subplots(figsize=(14, 6))
    metrics = ['accuracy', 'f1_macro', 'sensitivity', 'specificity']
    x = np.arange(len(results_df))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Model/Prompt Combination')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved metrics comparison to {output_dir}/metrics_comparison.png")

    # 3. Cost-effectiveness plot (if cost data available)
    if 'estimated_cost_usd' in results_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        scatter = ax.scatter(results_df['estimated_cost_usd'],
                            results_df['accuracy'],
                            s=200,
                            c=results_df['f1_macro'],
                            cmap='viridis',
                            alpha=0.6,
                            edgecolors='black')

        # Label points
        for idx, row in results_df.iterrows():
            ax.annotate(row['model'],
                       (row['estimated_cost_usd'], row['accuracy']),
                       fontsize=8,
                       ha='center')

        ax.set_xlabel('Estimated Cost (USD)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Cost vs. Accuracy (color = F1-score)')
        ax.grid(alpha=0.3)

        plt.colorbar(scatter, label='F1-Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cost_effectiveness.png', dpi=300, bbox_inches='tight')
        print(f"Saved cost-effectiveness plot to {output_dir}/cost_effectiveness.png")

    plt.close('all')

def print_summary(results_df):
    """
    Print a summary of results
    """
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print("\nTop 3 Models by Accuracy:")
    print(results_df[['model', 'accuracy', 'f1_macro', 'sensitivity', 'specificity']].head(3).to_string(index=False))

    if 'estimated_cost_usd' in results_df.columns:
        print("\n" + "="*80)
        print("COST ANALYSIS")
        print("="*80)

        print("\nTotal costs:")
        print(results_df[['model', 'total_tokens', 'estimated_cost_usd']].to_string(index=False))

        print("\n\nBest value for money (by accuracy per dollar):")
        best_value = results_df.nlargest(3, 'accuracy_per_dollar')
        print(best_value[['model', 'accuracy', 'estimated_cost_usd', 'accuracy_per_dollar']].to_string(index=False))

    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    # Run analysis
    results_df, predictions_df = analyze_results()

    # Create visualizations
    create_visualizations(results_df)

    # Print summary
    print_summary(results_df)

    # Save results to CSV
    results_df.to_csv('data/performance_metrics.csv', index=False)
    print("\n\nResults saved to data/performance_metrics.csv")
