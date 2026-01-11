"""
Evaluation script for Random Walk link prediction model.

Performs train/validation/test split, hyperparameter tuning, and comprehensive evaluation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import argparse
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from utils import load_graph
from RW_baseline.preprocessing import preprocess_graph
from RW_baseline.random_walk import generate_random_walks
from RW_baseline.embedding_learner import EmbeddingLearner
from RW_baseline.rw_prediction import RWPredictor


def split_edges_three_way(G, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split edges into train, validation, and test sets.
    
    Args:
        G: NetworkX graph
        train_ratio: Ratio of edges for training
        val_ratio: Ratio of edges for validation
        test_ratio: Ratio of edges for testing
        seed: Random seed
        
    Returns:
        G_train, val_edges, test_edges
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Get all unique edges
    all_edges = list(G.edges(data=True))
    random.shuffle(all_edges)
    
    # Calculate split points
    n_edges = len(all_edges)
    n_train = int(n_edges * train_ratio)
    n_val = int(n_edges * val_ratio)
    
    # Split edges
    train_edges = all_edges[:n_train]
    val_edges = all_edges[n_train:n_train + n_val]
    test_edges = all_edges[n_train + n_val:]
    
    # Create training graph
    import networkx as nx
    G_train = nx.Graph()
    
    # Add all nodes (preserve isolated nodes)
    G_train.add_nodes_from(G.nodes())
    
    # Add training edges
    for u, v, data in train_edges:
        G_train.add_edge(u, v, **data)
    
    # Convert to simple edge lists
    val_edge_list = [(u, v, data.get('weight', 1.0)) for u, v, data in val_edges]
    test_edge_list = [(u, v, data.get('weight', 1.0)) for u, v, data in test_edges]
    
    return G_train, val_edge_list, test_edge_list


def calculate_metrics(predictions, actual_edges, total_candidates):
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of (node1, node2, score) tuples
        actual_edges: Set of actual edges (as tuples)
        total_candidates: Total number of candidate edges evaluated
        
    Returns:
        Dictionary with precision, recall, f1, accuracy
    """
    if len(predictions) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': len(actual_edges),
            'true_negatives': total_candidates - len(actual_edges)
        }
    
    # Create set of predicted edges (canonical form)
    predicted_set = {tuple(sorted([u, v])) for u, v, _ in predictions}
    
    # Convert actual edges to canonical form
    actual_set = {tuple(sorted([u, v])) for u, v, _ in actual_edges}
    
    # Calculate confusion matrix
    true_positives = len(predicted_set & actual_set)
    false_positives = len(predicted_set - actual_set)
    false_negatives = len(actual_set - predicted_set)
    true_negatives = total_candidates - true_positives - false_positives - false_negatives
    
    # Calculate metrics
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = true_positives / len(actual_set) if len(actual_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total_candidates if total_candidates > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }


def evaluate_on_edges(predictor, G, actual_edges, threshold=0.5, top_k=None):
    """
    Evaluate predictor on a set of edges.
    
    Args:
        predictor: Trained RWPredictor
        G: Graph to evaluate on (training graph, not containing actual_edges)
        actual_edges: List of (node1, node2, weight) tuples to predict
        threshold: Prediction threshold
        top_k: Maximum number of predictions
        
    Returns:
        metrics, predictions
    """
    # Get all candidate pairs (non-existing edges in G)
    nodes = list(G.nodes())
    existing_edges = {tuple(sorted([u, v])) for u, v in G.edges()}
    
    all_candidates = []
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            edge = tuple(sorted([u, v]))
            if edge not in existing_edges:
                all_candidates.append((u, v))
    
    total_candidates = len(all_candidates)
    
    # Make predictions
    predictions = predictor.predict(all_candidates, method='hadamard', return_proba=True)
    
    # Filter by threshold
    predictions = [(u, v, score) for u, v, score in predictions if score >= threshold]
    
    # Apply top-k if specified
    if top_k is not None:
        predictions = predictions[:top_k]
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, actual_edges, total_candidates)
    
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Random Walk model with train/val/test split'
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
        '--dimensions',
        nargs='+',
        type=int,
        default=[128],
        help='Embedding dimensions to test (default: 128) - Higher is better but slower'
    )
    parser.add_argument(
        '--walk-length',
        nargs='+',
        type=int,
        default=[40, 80],
        help='Walk lengths to test (default: 40 80) - Longer captures more context'
    )
    parser.add_argument(
        '--num-walks',
        nargs='+',
        type=int,
        default=[10, 20],
        help='Number of walks per node to test (default: 10 20) - More walks = better embeddings'
    )
    parser.add_argument(
        '--threshold',
        nargs='+',
        type=float,
        default=[0.5, 0.6, 0.7],
        help='Prediction thresholds to test (default: 0.5 0.6 0.7) - Higher = more conservative'
    )
    parser.add_argument(
        '--top-k',
        nargs='+',
        type=int,
        default=[100, 200],
        help='Top-K values to test (default: 100 200) - Limit number of predictions'
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='logistic',
        choices=['logistic', 'rf'],
        help='Classifier type (default: logistic)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Random Walk Model - Evaluation with Train/Val/Test Split")
    print("="*60)
    
    # Load graph
    print(f"\nLoading graph from {args.input_file}...")
    G_full = load_graph(args.input_file)
    print(f"Total edges: {G_full.number_of_edges()}")
    print(f"Total nodes: {G_full.number_of_nodes()}")
    
    # Preprocess graph (connect isolated nodes)
    print("\n" + "="*60)
    print("PREPROCESSING")
    print("="*60)
    G_preprocessed, preprocess_stats = preprocess_graph(G_full, seed=args.seed)
    
    # Split data
    print(f"\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    
    G_train, val_edges, test_edges = split_edges_three_way(
        G_preprocessed, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Train edges: {G_train.number_of_edges()}")
    print(f"Validation edges: {len(val_edges)}")
    print(f"Test edges: {len(test_edges)}")
    
    # Hyperparameter tuning
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING ON VALIDATION SET")
    print("="*60)
    
    best_f1 = 0
    best_params = None
    best_embeddings = None
    results = []
    
    total_configs = (len(args.dimensions) * len(args.walk_length) * 
                    len(args.num_walks) * len(args.threshold) * (len(args.top_k) + 1))
    
    print(f"\nTesting {total_configs} configurations...")
    print(f"Dimensions: {args.dimensions}")
    print(f"Walk lengths: {args.walk_length}")
    print(f"Num walks: {args.num_walks}")
    print(f"Thresholds: {args.threshold}")
    print(f"Top-K: {args.top_k} + [None]")
    
    config_num = 0
    
    for dimensions in args.dimensions:
        for walk_length in args.walk_length:
            for num_walks in args.num_walks:
                # Generate random walks
                print(f"\n--- Generating walks (dim={dimensions}, len={walk_length}, num={num_walks}) ---")
                walks = generate_random_walks(
                    G_train,
                    num_walks=num_walks,
                    walk_length=walk_length,
                    p=1.0,
                    q=1.0,
                    seed=args.seed,
                    verbose=True
                )
                
                # Learn embeddings
                print(f"Learning embeddings...")
                learner = EmbeddingLearner(
                    dimensions=dimensions,
                    window=10,
                    epochs=10,
                    seed=args.seed
                )
                learner.train(walks, verbose=True)
                embeddings = learner.get_all_embeddings()
                
                # Train predictor
                print(f"Training predictor...")
                predictor = RWPredictor(
                    embeddings,
                    classifier_type=args.classifier,
                    random_state=args.seed
                )
                predictor.train(G_train, method='hadamard', verbose=True)
                
                # Evaluate with different thresholds and top-k
                for threshold in args.threshold:
                    for top_k in [None] + args.top_k:
                        config_num += 1
                        
                        print(f"\n[{config_num}/{total_configs}] Testing: "
                              f"dim={dimensions}, walk_len={walk_length}, num_walks={num_walks}, "
                              f"threshold={threshold}, top_k={top_k}")
                        
                        metrics, _ = evaluate_on_edges(
                            predictor, G_train, val_edges, threshold, top_k
                        )
                        
                        result = {
                            'dimensions': dimensions,
                            'walk_length': walk_length,
                            'num_walks': num_walks,
                            'threshold': threshold,
                            'top_k': top_k if top_k else 'all',
                            'precision': metrics['precision'],
                            'recall': metrics['recall'],
                            'f1': metrics['f1'],
                            'accuracy': metrics['accuracy']
                        }
                        results.append(result)
                        
                        print(f"  F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, "
                              f"Recall={metrics['recall']:.4f}")
                        
                        # Update best configuration
                        if metrics['f1'] > best_f1:
                            best_f1 = metrics['f1']
                            best_params = {
                                'dimensions': dimensions,
                                'walk_length': walk_length,
                                'num_walks': num_walks,
                                'threshold': threshold,
                                'top_k': top_k
                            }
                            best_embeddings = embeddings
                            print(f"  *** NEW BEST CONFIGURATION! F1={best_f1:.4f} ***")
    
    # Show results
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.sort_values('f1', ascending=False)
    
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS (Validation Set)")
    print("="*60)
    print(results_df_sorted.head(10).to_string(index=False))
    
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS (Validation F1={best_f1:.4f}):")
    print(f"  Dimensions: {best_params['dimensions']}")
    print(f"  Walk Length: {best_params['walk_length']}")
    print(f"  Num Walks: {best_params['num_walks']}")
    print(f"  Threshold: {best_params['threshold']}")
    print(f"  Top-K: {best_params['top_k'] if best_params['top_k'] else 'all'}")
    print(f"{'='*60}")
    
    # Evaluate on test set with best parameters
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Retrain with best parameters on training data
    print("\nRetraining with best parameters...")
    best_predictor = RWPredictor(
        best_embeddings,
        classifier_type=args.classifier,
        random_state=args.seed
    )
    best_predictor.train(G_train, method='hadamard', verbose=False)
    
    # Evaluate on test set
    test_metrics, test_predictions = evaluate_on_edges(
        best_predictor,
        G_train,
        test_edges,
        best_params['threshold'],
        best_params['top_k']
    )
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  Total predictions: {len(test_predictions)}")
    print(f"  True positives: {test_metrics['true_positives']}")
    print(f"  False positives: {test_metrics['false_positives']}")
    
    # Show example predictions
    test_edges_set = {tuple(sorted([u, v])) for u, v, _ in test_edges}
    
    print(f"\nTop 10 predictions on test set:")
    for i, (n1, n2, score) in enumerate(test_predictions[:10], 1):
        edge = tuple(sorted([n1, n2]))
        is_correct = edge in test_edges_set
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        print(f"  {i}. {n1} <-> {n2}: {score:.4f} [{status}]")
    
    # Save results
    output_dir = 'RW_baseline/output'
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = f'{output_dir}/validation_results.tsv'
    results_df_sorted.to_csv(results_file, sep='\t', index=False)
    print(f"\nValidation results saved to {results_file}")
    
    test_results_file = f'{output_dir}/test_predictions.tsv'
    test_df = pd.DataFrame([
        {
            'node1': n1,
            'node2': n2,
            'score': score,
            'is_correct': tuple(sorted([n1, n2])) in test_edges_set
        }
        for n1, n2, score in test_predictions
    ])
    test_df.to_csv(test_results_file, sep='\t', index=False)
    print(f"Test predictions saved to {test_results_file}")
    
    # Save evaluation summary
    summary_file = f'{output_dir}/evaluation_results.txt'
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RANDOM WALK MODEL - EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Best Hyperparameters (from validation set):\n")
        f.write(f"  Dimensions: {best_params['dimensions']}\n")
        f.write(f"  Walk Length: {best_params['walk_length']}\n")
        f.write(f"  Num Walks: {best_params['num_walks']}\n")
        f.write(f"  Threshold: {best_params['threshold']}\n")
        f.write(f"  Top-K: {best_params['top_k']}\n")
        f.write(f"  Validation F1: {best_f1:.4f}\n\n")
        f.write(f"Test Set Performance:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"  F1 Score: {test_metrics['f1']:.4f}\n")
    
    print(f"Evaluation summary saved to {summary_file}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == '__main__':
    main()
