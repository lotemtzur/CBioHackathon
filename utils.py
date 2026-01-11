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