"""
Random Walk generation module for graph link prediction.

This module implements Phase 1 of the Random Walk link prediction pipeline.
It generates weighted random walks on the graph using node2vec-style biased walks.
"""

import networkx as nx
import numpy as np
from typing import List, Tuple
import random


class WeightedRandomWalker:
    """
    Generates weighted random walks on a NetworkX graph.
    
    Implements node2vec-style biased walks with parameters:
    - p: Return parameter (likelihood of returning to previous node)
    - q: In-out parameter (BFS vs DFS exploration)
    
    Edge transition probabilities are weighted by edge weights.
    """
    
    def __init__(self, G: nx.Graph, p: float = 1.0, q: float = 1.0, seed: int = 42):
        """
        Initialize the random walker.
        
        Args:
            G: NetworkX graph with weighted edges
            p: Return parameter (default=1.0 for unbiased)
            q: In-out parameter (default=1.0 for unbiased)
            seed: Random seed for reproducibility
        """
        self.G = G
        self.p = p
        self.q = q
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Cache for transition probabilities
        self._transition_probs = {}
    
    def _get_edge_weight(self, u, v):
        """Get edge weight between nodes u and v."""
        if self.G.has_edge(u, v):
            return self.G[u][v].get('weight', 1.0)
        return 0.0
    
    def _calculate_transition_probs(self, current_node: str, previous_node: str = None) -> dict:
        """
        Calculate transition probabilities from current_node to its neighbors.
        
        Uses weighted edges and node2vec-style biased sampling if previous_node is given.
        
        Probability formula:
        - If no previous node: P(v) = Weight(current, v) / Σ_neighbors Weight
        - With node2vec bias:
            - If v == previous: weight / p
            - If v is neighbor of previous: weight
            - Otherwise: weight / q
        
        Args:
            current_node: Current position in walk
            previous_node: Previous position (for biased walks)
            
        Returns:
            Dictionary mapping neighbor -> probability
        """
        neighbors = list(self.G.neighbors(current_node))
        
        if not neighbors:
            return {}
        
        # Get edge weights
        weights = {}
        for neighbor in neighbors:
            base_weight = self._get_edge_weight(current_node, neighbor)
            
            if previous_node is None:
                # No bias, just use edge weight
                weights[neighbor] = base_weight
            else:
                # Apply node2vec bias
                if neighbor == previous_node:
                    # Return to previous node
                    weights[neighbor] = base_weight / self.p
                elif self.G.has_edge(neighbor, previous_node):
                    # Neighbor is also connected to previous (stay local)
                    weights[neighbor] = base_weight
                else:
                    # Explore further away
                    weights[neighbor] = base_weight / self.q
        
        # Normalize to probabilities
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            # Uniform probability if all weights are 0
            uniform_prob = 1.0 / len(neighbors)
            return {n: uniform_prob for n in neighbors}
        
        probs = {node: weight / total_weight for node, weight in weights.items()}
        
        return probs
    
    def generate_walk(self, start_node: str, walk_length: int) -> List[str]:
        """
        Generate a single random walk starting from start_node.
        
        Args:
            start_node: Starting node for the walk
            walk_length: Number of steps in the walk
            
        Returns:
            List of node IDs representing the walk path
        """
        walk = [start_node]
        
        current_node = start_node
        previous_node = None
        
        for _ in range(walk_length - 1):
            # Calculate transition probabilities
            probs = self._calculate_transition_probs(current_node, previous_node)
            
            if not probs:
                # Dead end - walk terminates early
                break
            
            # Sample next node
            neighbors = list(probs.keys())
            probabilities = list(probs.values())
            
            next_node = np.random.choice(neighbors, p=probabilities)
            
            # Update walk
            walk.append(next_node)
            previous_node = current_node
            current_node = next_node
        
        return walk
    
    def generate_walks(self, num_walks: int, walk_length: int, 
                       workers: int = 1, verbose: bool = True) -> List[List[str]]:
        """
        Generate multiple random walks from all nodes in the graph.
        
        Args:
            num_walks: Number of walks to generate from each node
            walk_length: Length of each walk
            workers: Number of parallel workers (not implemented, for compatibility)
            verbose: Whether to print progress
            
        Returns:
            List of walks, where each walk is a list of node IDs
        """
        nodes = list(self.G.nodes())
        walks = []
        
        if verbose:
            print(f"Generating {num_walks} walks of length {walk_length} from {len(nodes)} nodes...")
            print(f"Total walks to generate: {num_walks * len(nodes)}")
        
        for walk_iter in range(num_walks):
            if verbose and (walk_iter + 1) % 5 == 0:
                print(f"  Walk iteration {walk_iter + 1}/{num_walks}")
            
            # Shuffle nodes for each iteration
            shuffled_nodes = nodes.copy()
            random.shuffle(shuffled_nodes)
            
            for node in shuffled_nodes:
                walk = self.generate_walk(node, walk_length)
                walks.append(walk)
        
        if verbose:
            print(f"Generated {len(walks)} walks")
            avg_len = np.mean([len(w) for w in walks])
            print(f"Average walk length: {avg_len:.2f}")
        
        return walks


def generate_random_walks(G: nx.Graph, 
                         num_walks: int = 10,
                         walk_length: int = 80,
                         p: float = 1.0,
                         q: float = 1.0,
                         seed: int = 42,
                         verbose: bool = True) -> List[List[str]]:
    """
    Convenience function to generate random walks from a graph.
    
    Args:
        G: NetworkX graph with weighted edges
        num_walks: Number of walks per node
        walk_length: Length of each walk
        p: Return parameter for node2vec
        q: In-out parameter for node2vec
        seed: Random seed
        verbose: Print progress
        
    Returns:
        List of walks (each walk is a list of node IDs)
    """
    walker = WeightedRandomWalker(G, p=p, q=q, seed=seed)
    walks = walker.generate_walks(num_walks, walk_length, verbose=verbose)
    
    return walks
