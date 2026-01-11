"""
Diagnostic script to analyze Random Walk link prediction performance.

This script helps identify what's going wrong with the model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import networkx as nx
from utils import load_graph
from RW_baseline.preprocessing import preprocess_graph
from RW_baseline.random_walk import generate_random_walks
from RW_baseline.embedding_learner import EmbeddingLearner
from RW_baseline.rw_prediction import RWPredictor
import random


def analyze_embeddings(embeddings, G):
    """Analyze quality of learned embeddings."""
    print("\n=== EMBEDDING ANALYSIS ===")
    
    # Check embedding statistics
    all_embeddings = np.array(list(embeddings.values()))
    print(f"Embedding shape: {all_embeddings.shape}")
    print(f"Mean: {all_embeddings.mean():.4f}")
    print(f"Std: {all_embeddings.std():.4f}")
    print(f"Min: {all_embeddings.min():.4f}")
    print(f"Max: {all_embeddings.max():.4f}")
    
    # Check if embeddings are too similar (bad)
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(all_embeddings)
    np.fill_diagonal(similarities, 0)  # Ignore self-similarity
    
    print(f"\nPairwise Cosine Similarities:")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std: {similarities.std():.4f}")
    print(f"  Min: {similarities.min():.4f}")
    print(f"  Max: {similarities.max():.4f}")
    
    if similarities.mean() > 0.9:
        print("  ⚠️  WARNING: Embeddings are too similar! Model may not be learning distinctive features.")
    
    # Check embedding variance
    variances = all_embeddings.var(axis=0)
    low_variance_dims = np.sum(variances < 0.01)
    print(f"\nLow variance dimensions (<0.01): {low_variance_dims}/{len(variances)}")
    if low_variance_dims > len(variances) * 0.5:
        print("  ⚠️  WARNING: Too many low-variance dimensions! Embeddings may be collapsing.")


def analyze_predictions(predictor, G_train, test_edges):
    """Analyze prediction characteristics."""
    print("\n=== PREDICTION ANALYSIS ===")
    
    # Get all candidate pairs
    nodes = list(G_train.nodes())
    existing_edges = {tuple(sorted([u, v])) for u, v in G_train.edges()}
    
    candidates = []
    for i, u in enumerate(nodes[:50]):  # Sample first 50 nodes for speed
        for v in nodes[i+1:50]:
            edge = tuple(sorted([u, v]))
            if edge not in existing_edges:
                candidates.append((u, v))
    
    # Get predictions
    predictions = predictor.predict(candidates, method='hadamard', return_proba=True)
    
    # Analyze score distribution
    scores = [score for _, _, score in predictions]
    print(f"Prediction scores (sample of {len(scores)}):")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std: {np.std(scores):.4f}")
    print(f"  Min: {np.min(scores):.4f}")
    print(f"  Max: {np.max(scores):.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    
    # Check score distribution
    high_scores = np.sum(np.array(scores) > 0.8)
    med_scores = np.sum((np.array(scores) > 0.4) & (np.array(scores) <= 0.8))
    low_scores = np.sum(np.array(scores) <= 0.4)
    
    print(f"\nScore distribution:")
    print(f"  High (>0.8): {high_scores} ({high_scores/len(scores)*100:.1f}%)")
    print(f"  Medium (0.4-0.8): {med_scores} ({med_scores/len(scores)*100:.1f}%)")
    print(f"  Low (<0.4): {low_scores} ({low_scores/len(scores)*100:.1f}%)")
    
    if high_scores > len(scores) * 0.8:
        print("  ⚠️  WARNING: Most scores are high! Classifier may be too confident.")


def compare_with_common_neighbors(G_train, test_edges):
    """Compare with Common Neighbors baseline."""
    print("\n=== COMMON NEIGHBORS COMPARISON ===")
    
    test_edges_set = {tuple(sorted([u, v])) for u, v, _ in test_edges}
    
    # Calculate CN scores for test edges
    cn_scores = []
    for u, v, _ in test_edges[:20]:  # Sample 20 edges
        neighbors_u = set(G_train.neighbors(u)) if G_train.has_node(u) else set()
        neighbors_v = set(G_train.neighbors(v)) if G_train.has_node(v) else set()
        common = len(neighbors_u & neighbors_v)
        cn_scores.append(common)
    
    print(f"Common Neighbors for test edges (sample of 20):")
    print(f"  Mean: {np.mean(cn_scores):.2f}")
    print(f"  Edges with CN > 0: {np.sum(np.array(cn_scores) > 0)}/20")
    print(f"  Edges with CN = 0: {np.sum(np.array(cn_scores) == 0)}/20")
    
    if np.mean(cn_scores) < 1.0:
        print("  ℹ️  Many test edges have few/no common neighbors - this is a hard prediction task!")


def main():
    print("="*60)
    print("Random Walk Link Prediction - Diagnostics")
    print("="*60)
    
    # Load and split data
    print("\n1. Loading and splitting data...")
    graph_file = "string_interaction_physical_short.tsv"
    G = load_graph(graph_file)
    print(f"   Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G_processed, _ = preprocess_graph(G, seed=42)
    
    # Split
    all_edges = list(G_processed.edges(data=True))
    random.seed(42)
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * 0.8)
    train_edges = all_edges[:split_idx]
    test_edges = [(u, v, data.get('weight', 1.0)) for u, v, data in all_edges[split_idx:]]
    
    G_train = nx.Graph()
    G_train.add_nodes_from(G_processed.nodes())
    for u, v, data in train_edges:
        G_train.add_edge(u, v, **data)
    
    print(f"   Train: {G_train.number_of_edges()} edges, Test: {len(test_edges)} edges")
    
    # Test different configurations
    configs = [
        {"dims": 64, "walk_len": 20, "num_walks": 5, "name": "Original (Poor)"},
        {"dims": 128, "walk_len": 40, "num_walks": 10, "name": "Improved"},
        {"dims": 128, "walk_len": 80, "num_walks": 20, "name": "Best Effort"},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"  Dimensions: {config['dims']}, Walk Length: {config['walk_len']}, Num Walks: {config['num_walks']}")
        print(f"{'='*60}")
        
        # Generate walks
        print("\n2. Generating random walks...")
        walks = generate_random_walks(
            G_train,
            num_walks=config['num_walks'],
            walk_length=config['walk_len'],
            seed=42,
            verbose=False
        )
        print(f"   Generated {len(walks)} walks")
        
        # Learn embeddings
        print("\n3. Learning embeddings...")
        learner = EmbeddingLearner(
            dimensions=config['dims'],
            window=10,
            epochs=10,
            seed=42
        )
        learner.train(walks, verbose=False)
        embeddings = learner.get_all_embeddings()
        
        # Analyze embeddings
        analyze_embeddings(embeddings, G_train)
        
        # Train predictor
        print("\n4. Training predictor...")
        predictor = RWPredictor(embeddings, classifier_type='logistic', random_state=42)
        predictor.train(G_train, method='hadamard', verbose=False)
        
        # Analyze predictions
        analyze_predictions(predictor, G_train, test_edges)
        
        # Quick evaluation
        print("\n5. Quick evaluation...")
        from RW_baseline.evaluate_model import evaluate_on_edges
        metrics, _ = evaluate_on_edges(predictor, G_train, test_edges, threshold=0.5, top_k=100)
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1']:.4f}")
    
    # Compare with baseline
    compare_with_common_neighbors(G_train, test_edges)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("\n1. Try LONGER walks (80+) and MORE walks (20+)")
    print("2. Use HIGHER dimensions (128+)")
    print("3. Consider using Random Forest classifier instead of Logistic Regression")
    print("4. Try different feature combinations (average, weighted)")
    print("5. Add graph-based features (degree, clustering coefficient)")
    print("6. Consider hybrid approach: RW embeddings + Common Neighbors score")
    print("\nThe issue: The graph may be too sparse for pure random walk approaches!")


if __name__ == '__main__':
    main()
