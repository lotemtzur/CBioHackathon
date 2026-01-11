import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import random
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

def load_graph(file_path):
    """Load the protein interaction graph from TSV file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find header line
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            header_line = line[1:].strip()
            data_start = i + 1
            break
    
    if header_line is None:
        raise ValueError("No header line found")
    
    # Read data
    data = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(header_line + '\n' + data), sep='\t')
    
    # Get all edges (treating as undirected, so store both directions)
    edges = {}
    nodes = set()
    
    for _, row in df.iterrows():
        node1 = row['node1']
        node2 = row['node2']
        score = row['combined_score']
        
        # Store both directions
        edges[(node1, node2)] = score
        edges[(node2, node1)] = score
        
        nodes.add(node1)
        nodes.add(node2)
    
    return edges, nodes

def split_edges(edges, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split edges into train, validation, and test sets.
    Returns unique edges (each pair counted once).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get unique edges (only one direction per pair)
    unique_edges = []
    seen_pairs = set()
    
    for (n1, n2), score in edges.items():
        pair = tuple(sorted([n1, n2]))
        if pair not in seen_pairs:
            unique_edges.append((n1, n2, score))
            seen_pairs.add(pair)
    
    # Shuffle
    random.shuffle(unique_edges)
    
    # Split
    n_edges = len(unique_edges)
    n_train = int(n_edges * train_ratio)
    n_val = int(n_edges * val_ratio)
    
    train_edges = unique_edges[:n_train]
    val_edges = unique_edges[n_train:n_train + n_val]
    test_edges = unique_edges[n_train + n_val:]
    
    return train_edges, val_edges, test_edges

def build_adjacency(edges_list):
    """Build adjacency list from edge list."""
    adjacency = defaultdict(dict)
    edges_set = set()
    
    for n1, n2, score in edges_list:
        # Add both directions
        adjacency[n1][n2] = score
        adjacency[n2][n1] = score
        edges_set.add((n1, n2))
        edges_set.add((n2, n1))
    
    return adjacency, edges_set

def calculate_common_neighbor_score(node_x, node_y, adjacency):
    """Calculate common neighbor score."""
    neighbors_x = set(adjacency[node_x].keys())
    neighbors_y = set(adjacency[node_y].keys())
    
    common_neighbors = neighbors_x & neighbors_y
    
    total_score = 0.0
    for z in common_neighbors:
        score_zx = adjacency[node_x][z]
        score_zy = adjacency[node_y][z]
        total_score += min(score_zx, score_zy)
    
    return total_score

def predict_and_score(train_adjacency, train_edges_set, test_edges, all_nodes, 
                     threshold=0.5, top_k=None):
    """
    Predict edges on test set and calculate metrics.
    
    Returns:
        - precision: proportion of predicted edges that are correct
        - recall: proportion of test edges that were predicted
        - f1: harmonic mean of precision and recall
        - accuracy: proportion of all predictions (positive and negative) that are correct
        - predictions: list of (node1, node2, score, is_correct)
    """
    # Get test edges as set of tuples (both directions)
    test_edges_set = set()
    for n1, n2, _ in test_edges:
        test_edges_set.add((n1, n2))
        test_edges_set.add((n2, n1))
    
    # Get all possible node pairs not in training set
    node_list = list(all_nodes)
    all_candidate_pairs = []
    predictions = []
    
    for i, node_x in enumerate(node_list):
        for node_y in node_list[i+1:]:
            # Skip if already in training set
            if (node_x, node_y) in train_edges_set:
                continue
            
            score = calculate_common_neighbor_score(node_x, node_y, train_adjacency)
            
            # Check if this edge is in test set
            is_in_test = (node_x, node_y) in test_edges_set or (node_y, node_x) in test_edges_set
            all_candidate_pairs.append((node_x, node_y, score, is_in_test))
            
            if score > 0:
                predictions.append((node_x, node_y, score, is_in_test))
    
    # Sort by score
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Apply threshold
    filtered_predictions = [(n1, n2, s, c) for n1, n2, s, c in predictions if s >= threshold]
    
    # Apply top-k if specified
    if top_k is not None:
        filtered_predictions = filtered_predictions[:top_k]
    
    # Calculate metrics
    if len(filtered_predictions) == 0:
        return 0.0, 0.0, 0.0, 0.0, filtered_predictions
    
    # For filtered predictions
    true_positives = sum(1 for _, _, _, is_correct in filtered_predictions if is_correct)
    false_positives = len(filtered_predictions) - true_positives
    false_negatives = len(test_edges) - true_positives
    
    # True negatives: candidate pairs not predicted and not in test set
    predicted_pairs = set((n1, n2) for n1, n2, _, _ in filtered_predictions)
    true_negatives = sum(1 for n1, n2, _, is_in_test in all_candidate_pairs 
                        if (n1, n2) not in predicted_pairs and not is_in_test)
    
    precision = true_positives / len(filtered_predictions) if len(filtered_predictions) > 0 else 0
    recall = true_positives / len(test_edges) if len(test_edges) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    
    return precision, recall, f1, accuracy, filtered_predictions

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Common Neighbors model with train/val/test split'
    )
    parser.add_argument(
        'input_file',
        help='Input TSV file with protein interactions'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--threshold-range',
        nargs=3,
        type=float,
        default=[0.5, 5.0, 0.5],
        metavar=('MIN', 'MAX', 'STEP'),
        help='Threshold range to test: min max step (default: 0.5 5.0 0.5)'
    )
    parser.add_argument(
        '--top-k-values',
        nargs='+',
        type=int,
        default=[10, 20, 50, 100],
        help='Top-K values to test (default: 10 20 50 100)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Common Neighbors Model - Evaluation with Train/Val/Test Split")
    print("="*60)
    
    # Load full graph
    print(f"\nLoading graph from {args.input_file}...")
    all_edges, all_nodes = load_graph(args.input_file)
    print(f"Total unique edges: {len(all_edges) // 2}")
    print(f"Total nodes: {len(all_nodes)}")
    
    # Split data
    print(f"\nSplitting data (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})...")
    train_edges, val_edges, test_edges = split_edges(
        all_edges, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Train edges: {len(train_edges)}")
    print(f"Validation edges: {len(val_edges)}")
    print(f"Test edges: {len(test_edges)}")
    
    # Build training adjacency
    print("\nBuilding training graph...")
    train_adjacency, train_edges_set = build_adjacency(train_edges)
    
    # Hyperparameter tuning on validation set
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING ON VALIDATION SET")
    print("="*60)
    
    threshold_min, threshold_max, threshold_step = args.threshold_range
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    top_k_values = [None] + args.top_k_values  # None means no top-k limit
    
    best_f1 = 0
    best_params = None
    results = []
    
    print(f"\nTesting {len(thresholds)} thresholds × {len(top_k_values)} top-k values...")
    
    for threshold in thresholds:
        for top_k in top_k_values:
            precision, recall, f1, accuracy, _ = predict_and_score(
                train_adjacency, train_edges_set, val_edges, all_nodes, threshold, top_k
            )
            
            results.append({
                'threshold': threshold,
                'top_k': top_k if top_k else 'all',
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'threshold': threshold, 'top_k': top_k}
    
    # Show top 10 configurations
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values('f1', ascending=False)
    
    print("\nTop 10 configurations on validation set:")
    print(results_df_sorted.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS (Validation F1={best_f1:.4f}):")
    print(f"  Threshold: {best_params['threshold']}")
    print(f"  Top-K: {best_params['top_k'] if best_params['top_k'] else 'all'}")
    print(f"{'='*60}")
    
    # Evaluate on test set with best parameters
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    test_precision, test_recall, test_f1, test_accuracy, test_predictions = predict_and_score(
        train_adjacency, train_edges_set, test_edges, all_nodes,
        best_params['threshold'], best_params['top_k']
    )
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Total predictions: {len(test_predictions)}")
    print(f"  Correct predictions: {sum(1 for _, _, _, c in test_predictions if c)}")
    
    # Show some example predictions
    print(f"\nTop 10 predictions on test set:")
    for i, (n1, n2, score, is_correct) in enumerate(test_predictions[:10], 1):
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"  {i}. {n1} <-> {n2}: {score:.3f} [{status}]")
    
    # Save results
    results_file = 'validation_results.tsv'
    results_df_sorted.to_csv(results_file, sep='\t', index=False)
    print(f"\nValidation results saved to {results_file}")
    
    test_results_file = 'test_predictions.tsv'
    test_df = pd.DataFrame([
        {
            'node1': n1,
            'node2': n2,
            'score': score,
            'is_correct': is_correct,
            'actual_in_test': 'Yes' if is_correct else 'No'
        }
        for n1, n2, score, is_correct in test_predictions
    ])
    test_df.to_csv(test_results_file, sep='\t', index=False)
    print(f"Test predictions saved to {test_results_file}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    generate_plots(results_df_sorted, test_precision, test_recall, test_f1, test_accuracy,
                   test_predictions, best_params)
    print(f"Visualizations saved to output folder")

def generate_plots(results_df, test_precision, test_recall, test_f1, test_accuracy,
                   test_predictions, best_params):
    """Generate visualization plots for the evaluation results."""
    
    # Set style
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('output', exist_ok=True)
    
    # 1. Hyperparameter Heatmap (Threshold vs Top-K)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap - separate 'all' and numeric top_k
    results_numeric = results_df[results_df['top_k'] != 'all'].copy()
    results_all = results_df[results_df['top_k'] == 'all'].copy()
    
    if len(results_numeric) > 0:
        # Convert top_k to numeric
        results_numeric['top_k'] = pd.to_numeric(results_numeric['top_k'])
        pivot_table = results_numeric.pivot_table(
            values='f1', 
            index='top_k', 
            columns='threshold', 
            aggfunc='mean'
        )
        
        # Use seaborn if available, otherwise matplotlib
        if HAS_SEABORN:
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'F1 Score'})
        else:
            im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(np.arange(len(pivot_table.columns)))
            ax.set_yticks(np.arange(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns)
            ax.set_yticklabels(pivot_table.index)
            
            # Add text annotations
            for i in range(len(pivot_table.index)):
                for j in range(len(pivot_table.columns)):
                    text = ax.text(j, i, f'{pivot_table.values[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=ax, label='F1 Score')
        
        ax.set_title('Hyperparameter Tuning: F1 Score Heatmap\n(Threshold vs Top-K)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Threshold', fontsize=12)
        ax.set_ylabel('Top-K', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No numeric top-k values to plot', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Hyperparameter Tuning: F1 Score Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/hyperparameter_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Metrics Comparison (Accuracy, Precision, Recall, F1)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [test_accuracy, test_precision, test_recall, test_f1]
    colors = ['#f39c12', '#3498db', '#e74c3c', '#2ecc71']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Configurations by F1 Score
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_10 = results_df.head(10).copy()
    top_10['config'] = top_10.apply(
        lambda x: f"T={x['threshold']}, K={x['top_k']}", axis=1
    )
    
    y_pos = np.arange(len(top_10))
    ax.barh(y_pos, top_10['f1'], color='#9b59b6', alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_10['config'])
    ax.invert_yaxis()
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Top 10 Hyperparameter Configurations (Validation Set)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(top_10.iterrows()):
        ax.text(row['f1'], i, f" {row['f1']:.4f}", 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/top_configurations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Precision-Recall Trade-off
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all configurations
    ax.scatter(results_df['recall'], results_df['precision'], 
              alpha=0.5, s=100, c=results_df['f1'], cmap='viridis', edgecolors='black')
    
    # Highlight best configuration
    best_config = results_df.iloc[0]
    ax.scatter(best_config['recall'], best_config['precision'], 
              s=300, c='red', marker='*', edgecolors='black', linewidth=2,
              label=f"Best Config (F1={best_config['f1']:.3f})", zorder=5)
    
    # Mark test set performance
    ax.scatter(test_recall, test_precision, 
              s=300, c='green', marker='D', edgecolors='black', linewidth=2,
              label=f"Test Set (F1={test_f1:.3f})", zorder=5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off\n(All Hyperparameter Configurations)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                               norm=plt.Normalize(vmin=results_df['f1'].min(), 
                                                 vmax=results_df['f1'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('F1 Score', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('output/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Score Distribution of Predictions
    if len(test_predictions) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Separate correct and incorrect predictions
        correct_scores = [score for _, _, score, is_correct in test_predictions if is_correct]
        incorrect_scores = [score for _, _, score, is_correct in test_predictions if not is_correct]
        
        # Histogram
        bins = np.linspace(
            min([s for _, _, s, _ in test_predictions]),
            max([s for _, _, s, _ in test_predictions]),
            30
        )
        
        ax1.hist(correct_scores, bins=bins, alpha=0.7, label='Correct', 
                color='green', edgecolor='black')
        ax1.hist(incorrect_scores, bins=bins, alpha=0.7, label='Incorrect', 
                color='red', edgecolor='black')
        ax1.set_xlabel('Common Neighbor Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Score Distribution of Predictions', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [correct_scores, incorrect_scores]
        box = ax2.boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                          patch_artist=True, showmeans=True)
        
        # Color the boxes
        colors_box = ['lightgreen', 'lightcoral']
        for patch, color in zip(box['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Common Neighbor Score', fontsize=12)
        ax2.set_title('Score Distribution by Correctness', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('output/score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 6. ROC and Precision-Recall Curves (AUC)
    if len(test_predictions) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Get scores and labels
        all_scores = [score for _, _, score, _ in test_predictions]
        all_labels = [1 if is_correct else 0 for _, _, _, is_correct in test_predictions]
        
        # Sort by score (descending)
        sorted_pairs = sorted(zip(all_scores, all_labels), key=lambda x: x[0], reverse=True)
        sorted_scores, sorted_labels = zip(*sorted_pairs)
        
        # Calculate cumulative metrics for different thresholds
        tpr_list = []  # True Positive Rate (Recall)
        fpr_list = []  # False Positive Rate
        precision_list = []
        recall_list = []
        
        total_positives = sum(sorted_labels)
        total_negatives = len(sorted_labels) - total_positives
        
        for i in range(len(sorted_scores) + 1):
            if i == 0:
                # No predictions (threshold = infinity)
                tp, fp = 0, 0
            else:
                # Predictions above threshold = first i items
                tp = sum(sorted_labels[:i])
                fp = i - tp
            
            # Calculate rates
            tpr = tp / total_positives if total_positives > 0 else 0
            fpr = fp / total_negatives if total_negatives > 0 else 0
            precision = tp / i if i > 0 else 1.0
            recall = tpr
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            precision_list.append(precision)
            recall_list.append(recall)
        
        # Calculate AUC using trapezoidal rule
        roc_auc = 0
        for i in range(1, len(fpr_list)):
            roc_auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
        
        pr_auc = 0
        for i in range(1, len(recall_list)):
            pr_auc += (recall_list[i-1] - recall_list[i]) * (precision_list[i] + precision_list[i-1]) / 2
        
        # Plot ROC Curve
        ax1.plot(fpr_list, tpr_list, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot Precision-Recall Curve
        ax2.plot(recall_list, precision_list, color='purple', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        
        # Add baseline (random classifier)
        baseline = total_positives / len(sorted_labels)
        ax2.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                   label=f'Random classifier (precision = {baseline:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax2.legend(loc="upper right", fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/auc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 7. Summary Statistics Figure
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Common Neighbors Model - Evaluation Summary', 
                 fontsize=16, fontweight='bold')
    
    # Create text summary
    summary_text = f"""
    BEST HYPERPARAMETERS (Selected on Validation Set):
    • Threshold: {best_params['threshold']}
    • Top-K: {best_params['top_k'] if best_params['top_k'] else 'all'}
    
    TEST SET PERFORMANCE:
    • Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)
    • Precision: {test_precision:.4f} ({test_precision*100:.2f}%)
    • Recall: {test_recall:.4f} ({test_recall*100:.2f}%)
    • F1 Score: {test_f1:.4f}
    
    PREDICTION STATISTICS:
    • Total Predictions: {len(test_predictions)}
    • Correct Predictions: {sum(1 for _, _, _, c in test_predictions if c)}
    • Incorrect Predictions: {sum(1 for _, _, _, c in test_predictions if not c)}
    
    INTERPRETATION:
    • Accuracy: {test_accuracy*100:.1f}% of all predictions (pos+neg) correct
    • Precision: {test_precision*100:.1f}% of predicted edges were correct
    • Recall: Found {test_recall*100:.1f}% of all actual interactions
    • F1 score balances precision and recall
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.5))
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
