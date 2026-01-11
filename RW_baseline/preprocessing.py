"""
Preprocessing module for handling isolated nodes in the graph.

This module implements Phase 0 of the Random Walk link prediction pipeline.
It ensures all nodes are connected by adding synthetic edges to isolated nodes
based on weighted degree (node strength) probabilities.
"""

import networkx as nx
import numpy as np
from typing import Tuple


def calculate_node_strength(G: nx.Graph) -> dict:
    """
    Calculate the weighted degree (node strength) for all connected nodes.
    
    For each node i, calculate: S_i = Σ_j Weight(i,j) for all neighbors j
    
    Args:
        G: NetworkX graph with weighted edges
        
    Returns:
        Dictionary mapping node -> strength
    """
    node_strength = {}
    
    for node in G.nodes():
        # Sum all edge weights for this node
        strength = sum(G[node][neighbor].get('weight', 1.0) 
                      for neighbor in G.neighbors(node))
        node_strength[node] = strength
    
    return node_strength


def calculate_attachment_probabilities(node_strength: dict) -> dict:
    """
    Calculate attachment probability for each node based on its strength.
    
    P_i = S_i / Σ_all_nodes S
    
    Args:
        node_strength: Dictionary of node -> strength values
        
    Returns:
        Dictionary mapping node -> attachment probability
    """
    total_strength = sum(node_strength.values())
    
    if total_strength == 0:
        # If all nodes have 0 strength, use uniform probability
        num_nodes = len(node_strength)
        return {node: 1.0/num_nodes for node in node_strength}
    
    attachment_probs = {
        node: strength / total_strength 
        for node, strength in node_strength.items()
    }
    
    return attachment_probs


def connect_isolated_nodes(G: nx.Graph, seed: int = 42) -> Tuple[nx.Graph, int]:
    """
    Connect isolated nodes to the main graph using preferential attachment.
    
    For each isolated node:
    1. Calculate attachment probabilities based on node strength
    2. Randomly select a neighbor from connected nodes
    3. Add edge with weight = average graph weight
    
    Args:
        G: NetworkX graph (may contain isolated nodes)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (connected_graph, num_isolated_nodes_connected)
    """
    np.random.seed(seed)
    
    # Create a copy to avoid modifying the original
    G_connected = G.copy()
    
    # Identify isolated nodes (degree = 0)
    isolated_nodes = list(nx.isolates(G_connected))
    
    if not isolated_nodes:
        print("No isolated nodes found. Graph is already connected.")
        return G_connected, 0
    
    # Get connected nodes (non-isolated)
    connected_nodes = [node for node in G_connected.nodes() if node not in isolated_nodes]
    
    if not connected_nodes:
        print("WARNING: All nodes are isolated! Cannot connect.")
        return G_connected, 0
    
    # Calculate node strength for connected nodes
    node_strength = calculate_node_strength(G_connected)
    
    # Filter to only connected nodes
    connected_strength = {node: strength for node, strength in node_strength.items() 
                         if node in connected_nodes}
    
    # Calculate attachment probabilities
    attachment_probs = calculate_attachment_probabilities(connected_strength)
    
    # Calculate average edge weight
    if G_connected.number_of_edges() > 0:
        avg_weight = np.mean([data.get('weight', 1.0) 
                             for _, _, data in G_connected.edges(data=True)])
    else:
        avg_weight = 1.0
    
    # Connect each isolated node
    prob_nodes = list(attachment_probs.keys())
    prob_values = list(attachment_probs.values())
    
    for isolated_node in isolated_nodes:
        # Select a random connected node based on attachment probabilities
        selected_node = np.random.choice(prob_nodes, p=prob_values)
        
        # Add synthetic edge with average weight
        G_connected.add_edge(isolated_node, selected_node, weight=avg_weight)
        
        print(f"Connected isolated node {isolated_node} to {selected_node} "
              f"with weight {avg_weight:.4f}")
    
    print(f"\nTotal isolated nodes connected: {len(isolated_nodes)}")
    
    return G_connected, len(isolated_nodes)


def preprocess_graph(G: nx.Graph, seed: int = 42) -> Tuple[nx.Graph, dict]:
    """
    Full preprocessing pipeline for the graph.
    
    Args:
        G: Input graph
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (preprocessed_graph, preprocessing_stats)
    """
    stats = {
        'original_nodes': G.number_of_nodes(),
        'original_edges': G.number_of_edges(),
        'isolated_nodes_connected': 0
    }
    
    # Connect isolated nodes
    G_processed, num_isolated = connect_isolated_nodes(G, seed=seed)
    stats['isolated_nodes_connected'] = num_isolated
    
    stats['final_nodes'] = G_processed.number_of_nodes()
    stats['final_edges'] = G_processed.number_of_edges()
    
    print("\n=== Preprocessing Statistics ===")
    print(f"Original nodes: {stats['original_nodes']}")
    print(f"Original edges: {stats['original_edges']}")
    print(f"Isolated nodes connected: {stats['isolated_nodes_connected']}")
    print(f"Final nodes: {stats['final_nodes']}")
    print(f"Final edges: {stats['final_edges']}")
    
    return G_processed, stats
