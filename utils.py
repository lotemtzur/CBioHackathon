import networkx as nx
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import combinations
import numpy as np


def load_graph(file_path="string_interactions_short.tsv"):
    """Load a weighted graph from TSV file into NetworkX using combined_score as weights."""
    # Read TSV (first line is header with #)
    df = pd.read_csv(file_path, sep='\t')
    
    # Remove '#' from first column name if present
    df.columns = df.columns.str.lstrip('#')
    
    # Create weighted graph
    G = nx.Graph()
    
    # Add edges with combined_score as weight
    for _, row in df.iterrows():
        G.add_edge(row['node1'], row['node2'], weight=row['combined_score'])
    
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G


def compare_predictions(predictions, test_edges, non_train_edges):
    tp = fp = tn = fn = 0
    
    test_set = set()
    for u, v, _ in test_edges:
        test_set.add(tuple(sorted((u, v))))
        
    for (u, v) in non_train_edges:
        pred = predictions.get((u, v), predictions.get((v, u), False))
        
        actual = tuple(sorted((u, v))) in test_set
        
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        elif not pred and actual:
            fn += 1
            
    return tp, fp, tn, fn

    
def calculate_metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def plot_roc_curve(predictions_dict, test_edges, non_train_edges, model_name=None):
    """
    Correct AUC calculation for Link Prediction.
    Only evaluates on non-training edges (test edges + other negative samples).
    """
    y_true = []
    y_scores = []
    
    # Create sorted set for O(1) lookup
    test_set = set([tuple(sorted((u, v))) for u, v, _ in test_edges])
    non_train_set = set([tuple(sorted((u, v))) for u, v in non_train_edges])
    
    # Only evaluate on non-training edges
    for (u, v), score in predictions_dict.items():
        edge = tuple(sorted((u, v)))
        if edge in non_train_set:  # Only evaluate on held-out edges
            y_true.append(1 if edge in test_set else 0)
            y_scores.append(score)
    
    if len(set(y_true)) < 2:
        print("Error: Not enough classes for AUC.")
        return None
    
    auc_score = roc_auc_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Link Prediction ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(f'roc_{model_name}.png')
    plt.show()
    
    return auc_score

# ============ Train Test Split Functions - Helpers =============

def _get_scores(G, split_type):
    """Common scoring logic for both edges and vertices."""
    degrees = dict(G.degree(weight='weight'))
    if split_type == 'edge':
        items = list(G.edges(data=True))
        scores = np.array([(degrees[u] + degrees[v]) / 2 for u, v, _ in items])
    else:  # vertex
        items = list(G.nodes())
        scores = np.array([degrees[node] for node in items])
    return items, scores

def _finalize_split(G, test_units, split_type):
    """Common logic to reconstruct G_train and non_train_edges."""
    test_edges = []
    
    # Identify which edges belong in the test set
    if split_type == 'edge':
        # Compare only (u, v), not the unhashable data dict
        test_units_set = {tuple(sorted((u, v))) for u, v, _ in test_units}
        for u, v, d in G.edges(data=True):
            if tuple(sorted((u, v))) in test_units_set:
                test_edges.append((u, v, d))
    else:  # vertex split
        test_units_set = set(test_units)
        for u, v, d in G.edges(data=True):
            if u in test_units_set or v in test_units_set:
                test_edges.append((u, v, d))

    # Build G_train
    G_train = G.copy()
    G_train.remove_edges_from([(u, v) for u, v, _ in test_edges])
    
    # Calculate negative space (everything not in G_train)
    train_edges_set = {tuple(sorted((u, v))) for u, v in G_train.edges()}
    nodes = list(G.nodes())
    non_train_edges = [
        pair for pair in combinations(nodes, 2) 
        if tuple(sorted(pair)) not in train_edges_set
    ]
    
    return G_train, test_edges, non_train_edges

# ============ Train Test Split Functions =============

def graph_split_rnd(G, test_ratio=0.2, split_type='edge', rnd_seed=42):
    random.seed(rnd_seed)
    items, _ = _get_scores(G, split_type)
    
    # Simple random shuffle selection
    random.shuffle(items)
    n_test = int(len(items) * test_ratio)
    test_units = items[:n_test]
    
    return _finalize_split(G, test_units, split_type)

def density_biased_split(G, test_ratio=0.2, side='high', split_type='edge'):
    """side: 'high' or 'low' density prioritization for test set."""
    items, scores = _get_scores(G, split_type)
    
    # Sort based on density
    sorted_idx = np.argsort(scores)
    if side == 'high':
        sorted_idx = sorted_idx[::-1]
    
    n_test = int(len(items) * test_ratio)
    test_units = [items[i] for i in sorted_idx[:n_test]]
    
    return _finalize_split(G, test_units, split_type)

def preserve_density_split(G, test_ratio=0.2, split_type="edge", n_bins=10, rnd_seed=42):
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    items, scores = _get_scores(G, split_type)
    
    # Stratified binning selection
    bins = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    test_units = []
    
    for i in range(len(bins) - 1):
        mask = (scores >= bins[i]) & (scores <= bins[i+1])
        bin_indices = np.where(mask)[0].tolist()
        random.shuffle(bin_indices)
        
        n_bin_test = int(len(bin_indices) * test_ratio)
        test_units.extend([items[idx] for idx in bin_indices[:n_bin_test]])
            
    return _finalize_split(G, test_units, split_type)

# ============ End of Train Test Split Functions =============

# Alon
def structure_preserving_edge_split(G, test_ratio=0.2, seed=42):
    """
    Splits graph edges into Train and Test while trying to preserve
    graph connectivity (preventing isolated nodes).
    
    Args:
        G (nx.Graph): The original graph.
        test_ratio (float): Fraction of edges to hide for testing.
        seed (int): Random seed.
        
    Returns:
        G_train (nx.Graph): Graph with all nodes but only training edges.
        test_edges (list): The hidden edges (Positive samples for test).
        neg_test_edges (list): Non-existent edges (Negative samples for test).
    """
    random.seed(seed)
    
    # 1. Start with a full copy of the graph
    G_train = G.copy()
    nodes = list(G.nodes())
    edges = list(G.edges())
    random.shuffle(edges)
    
    num_test = int(len(edges) * test_ratio)
    test_edges = []
    
    # 2. Select Test Edges INTELLIGENTLY
    # We iterate through random edges and remove them ONLY IF
    # removing them doesn't isolate a node.
    
    print(f"Selecting {num_test} edges for testing...")
    
    for u, v in edges:
        # Stop if we have enough test edges
        if len(test_edges) >= num_test:
            break
            
        # Check degrees in the current training graph
        # We only remove edge (u,v) if both u and v have other connections
        if G_train.degree(u) > 1 and G_train.degree(v) > 1:
            G_train.remove_edge(u, v)
            test_edges.append((u, v))
            
    # Note: If the graph is very sparse, we might not reach the full test_ratio
    # without isolating nodes. This logic prioritizes structure over exact ratio.
    print(f"Actual Test Edges selected: {len(test_edges)} (Structure Preserved)")

    # 3. Generate Negative Edges (Edges that don't exist)
    # We need them to test if the model can distinguish True from False
    neg_test_edges = set()
    
    # Heuristic: Try to get same amount of negatives as positives
    target_neg = len(test_edges)
    
    while len(neg_test_edges) < target_neg:
        # Random sampling is much faster than combinations() for large graphs
        u = random.choice(nodes)
        v = random.choice(nodes)
        
        if u != v and not G.has_edge(u, v):
            # Sort tuple to avoid (u,v) vs (v,u) duplicates
            edge = tuple(sorted((u, v)))
            neg_test_edges.add(edge)
            
    return G_train, test_edges, list(neg_test_edges)

