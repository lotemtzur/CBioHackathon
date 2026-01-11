import networkx as nx
import pandas as pd
import random
import sys


def load_graph(file_path="string_interaction_physical.tsv"):
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
    """Split the graph into training and testing sets by removing a fraction of edges."""
    random.seed(rnd_seed)
    edges = list(G.edges(data=True))
    random.shuffle(edges) # TODO: Maybe use other type of shuffling 
    
    n_test = int(len(edges) * test_ratio)
    test_edges = edges[:n_test]
    train_edges = edges[n_test:]
    
    G_train = G.copy()
    G_train.remove_edges_from([(u, v) for u, v, _ in test_edges])
    
    return G_train, test_edges


def compare_predictions(predictions, test_edges):
    """
    predictions comparison:
    Count TP, FP, TN, FN based on predictions and actual test edges.
    predications: Dict with keys as (u,v) tuples.
    test_edges: List of (u,v,weight) tuples representing actual edges.
    """
    tp = fp = tn = fn = 0
    for (u, v, weight) in test_edges:
        pred = predictions.get((u, v), predictions.get((v, u), False))
        actual = True  # Since these are test edges, they exist
        
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