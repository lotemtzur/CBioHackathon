"""
Quick example script demonstrating Random Walk link prediction.

This script shows a minimal working example with the short dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_graph
from RW_baseline.preprocessing import preprocess_graph
from RW_baseline.random_walk import generate_random_walks
from RW_baseline.embedding_learner import EmbeddingLearner
from RW_baseline.rw_prediction import RWPredictor
import networkx as nx


def main():
    print("="*60)
    print("Random Walk Link Prediction - Quick Example")
    print("="*60)
    
    # Load graph
    print("\n1. Loading graph...")
    graph_file = "string_interaction_physical_short.tsv"
    G = load_graph(graph_file)
    print(f"   Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Preprocess
    print("\n2. Preprocessing (connecting isolated nodes)...")
    G_processed, stats = preprocess_graph(G, seed=42)
    
    # Simple train/test split (80/20)
    print("\n3. Splitting data (80% train, 20% test)...")
    all_edges = list(G_processed.edges(data=True))
    import random
    random.seed(42)
    random.shuffle(all_edges)
    
    split_idx = int(len(all_edges) * 0.8)
    train_edges = all_edges[:split_idx]
    test_edges = all_edges[split_idx:]
    
    # Create training graph
    G_train = nx.Graph()
    G_train.add_nodes_from(G_processed.nodes())
    for u, v, data in train_edges:
        G_train.add_edge(u, v, **data)
    
    print(f"   Train: {G_train.number_of_edges()} edges")
    print(f"   Test: {len(test_edges)} edges")
    
    # Generate random walks
    print("\n4. Generating random walks...")
    walks = generate_random_walks(
        G_train,
        num_walks=10,
        walk_length=40,
        p=1.0,
        q=1.0,
        seed=42,
        verbose=True
    )
    
    # Learn embeddings
    print("\n5. Learning node embeddings...")
    learner = EmbeddingLearner(
        dimensions=64,
        window=10,
        epochs=10,
        seed=42
    )
    learner.train(walks, verbose=True)
    embeddings = learner.get_all_embeddings()
    
    # Train predictor
    print("\n6. Training link predictor...")
    predictor = RWPredictor(
        embeddings,
        classifier_type='logistic',
        random_state=42
    )
    predictor.train(G_train, method='hadamard', verbose=True)
    
    # Predict new edges
    print("\n7. Predicting new edges...")
    predictions = predictor.predict_new_edges(
        G_train,
        top_k=10,
        threshold=0.5,
        method='hadamard',
        verbose=True
    )
    
    print("\n8. Top 10 Predictions:")
    print("-" * 60)
    
    test_edges_set = {tuple(sorted([u, v])) for u, v, _ in test_edges}
    
    for i, (u, v, score) in enumerate(predictions[:10], 1):
        edge = tuple(sorted([u, v]))
        is_correct = edge in test_edges_set
        status = "✓ IN TEST SET" if is_correct else "✗ NOT IN TEST SET"
        print(f"{i:2d}. {u:15s} <-> {v:15s} | Score: {score:.4f} | {status}")
    
    # Calculate simple accuracy
    num_correct = sum(1 for u, v, _ in predictions 
                     if tuple(sorted([u, v])) in test_edges_set)
    
    print("\n" + "="*60)
    print(f"Accuracy: {num_correct}/{len(predictions)} = {num_correct/len(predictions)*100:.1f}%")
    print("="*60)
    
    print("\n✅ Example completed successfully!")
    print("\nTo run full evaluation with hyperparameter tuning:")
    print("  python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv")


if __name__ == '__main__':
    main()
