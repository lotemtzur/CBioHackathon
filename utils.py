import networkx as nx
import pandas as pd
import random
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from itertools import combinations


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


def graph_train_test_split(G, test_ratio=0.2, rnd_seed=42):
    random.seed(rnd_seed)
    nodes = list(G.nodes())
    edges = list(G.edges(data=True))
    random.shuffle(edges)
    
    n_test = int(len(edges) * test_ratio)
    test_edges = edges[:n_test]
    train_edges = edges[n_test:]

    G_train = G.copy()
    G_train.remove_edges_from([(u, v) for u, v, _ in test_edges])
    
    train_edges_set = set()
    for u, v, _ in train_edges:
        train_edges_set.add(tuple(sorted((u, v))))
        
    non_train_edges = []
    for u, v in combinations(nodes, 2):
        if tuple(sorted((u, v))) not in train_edges_set:
            non_train_edges.append((u, v))
            
    return G_train, test_edges, non_train_edges


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
