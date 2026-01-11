"""
Hybrid predictor combining Random Walk embeddings with Common Neighbors.

This approach leverages both structural embeddings and local topology.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random


class HybridRWPredictor:
    """
    Hybrid link prediction using:
    1. Random Walk embeddings (Hadamard product)
    2. Common Neighbors score
    3. Weighted degree features
    """
    
    def __init__(self, embeddings: Dict[str, np.ndarray], G: nx.Graph,
                 classifier_type: str = 'rf',
                 random_state: int = 42):
        """
        Initialize the hybrid predictor.
        
        Args:
            embeddings: Dictionary mapping node ID -> embedding vector
            G: Graph for computing topological features
            classifier_type: 'logistic' or 'rf'
            random_state: Random seed
        """
        self.embeddings = embeddings
        self.G = G
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.classifier = None
        
        # Get embedding dimension
        first_embedding = next(iter(embeddings.values()))
        self.embedding_dim = len(first_embedding)
    
    def _compute_edge_features(self, node1: str, node2: str) -> np.ndarray:
        """
        Compute comprehensive edge features.
        
        Features include:
        - Hadamard product of embeddings
        - Common neighbors count
        - Weighted common neighbors score
        - Sum of degrees
        - Product of degrees
        - Preferential attachment score
        """
        # Get embeddings
        emb1 = self.embeddings.get(str(node1), np.zeros(self.embedding_dim))
        emb2 = self.embeddings.get(str(node2), np.zeros(self.embedding_dim))
        
        # Hadamard product
        hadamard = emb1 * emb2
        
        # Topological features
        neighbors1 = set(self.G.neighbors(node1)) if self.G.has_node(node1) else set()
        neighbors2 = set(self.G.neighbors(node2)) if self.G.has_node(node2) else set()
        common_neighbors = neighbors1 & neighbors2
        
        # Common neighbors count
        cn_count = len(common_neighbors)
        
        # Weighted common neighbors (Adamic-Adar inspired)
        wcn_score = 0.0
        for neighbor in common_neighbors:
            neighbor_degree = self.G.degree(neighbor)
            if neighbor_degree > 1:
                wcn_score += 1.0 / np.log(neighbor_degree)
        
        # Degree features
        degree1 = self.G.degree(node1) if self.G.has_node(node1) else 0
        degree2 = self.G.degree(node2) if self.G.has_node(node2) else 0
        
        # Preferential attachment
        pref_attach = degree1 * degree2
        
        # Jaccard coefficient
        union_size = len(neighbors1 | neighbors2)
        jaccard = cn_count / union_size if union_size > 0 else 0.0
        
        # Combine all features
        topo_features = np.array([
            cn_count,
            wcn_score,
            degree1 + degree2,
            pref_attach,
            jaccard
        ])
        
        # Concatenate embeddings and topological features
        features = np.concatenate([hadamard, topo_features])
        
        return features
    
    def _generate_negative_samples(self, num_samples: int,
                                   existing_edges: set) -> List[Tuple[str, str]]:
        """Generate negative samples (non-existing edges)."""
        nodes = list(self.G.nodes())
        negative_samples = []
        
        max_attempts = num_samples * 10
        attempts = 0
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            node1, node2 = random.sample(nodes, 2)
            edge = tuple(sorted([node1, node2]))
            
            if edge not in existing_edges and not self.G.has_edge(node1, node2):
                negative_samples.append(edge)
            
            attempts += 1
        
        return negative_samples
    
    def train(self, positive_edges: List[Tuple], balance_ratio: float = 1.0, 
              verbose: bool = True) -> None:
        """
        Train the hybrid classifier.
        
        Args:
            positive_edges: List of edges from training graph
            balance_ratio: Ratio of negative to positive samples
            verbose: Print progress
        """
        if verbose:
            print("\n=== Training Hybrid Link Predictor ===")
            print(f"Classifier type: {self.classifier_type}")
            print(f"Embedding dims: {self.embedding_dim}")
            print(f"Topological features: 5")
            print(f"Total feature dims: {self.embedding_dim + 5}")
        
        num_positive = len(positive_edges)
        if verbose:
            print(f"Positive samples: {num_positive}")
        
        # Create set of existing edges
        existing_edges = {tuple(sorted([u, v])) for u, v in positive_edges}
        
        # Generate negative samples
        num_negative = int(num_positive * balance_ratio)
        negative_edges = self._generate_negative_samples(num_negative, existing_edges)
        
        if verbose:
            print(f"Negative samples: {len(negative_edges)}")
        
        # Prepare training data
        X = []
        y = []
        
        # Add positive samples
        for u, v in positive_edges:
            features = self._compute_edge_features(u, v)
            X.append(features)
            y.append(1)
        
        # Add negative samples
        for u, v in negative_edges:
            features = self._compute_edge_features(u, v)
            X.append(features)
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"Training data shape: {X.shape}")
        
        # Train classifier
        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        
        self.classifier.fit(X, y)
        
        if verbose:
            train_score = self.classifier.score(X, y)
            print(f"Training accuracy: {train_score:.4f}")
    
    def predict(self, node_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
        """
        Predict edge probabilities.
        
        Args:
            node_pairs: List of (node1, node2) tuples
            
        Returns:
            List of (node1, node2, probability) tuples sorted by probability
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Compute features
        X = []
        for u, v in node_pairs:
            features = self._compute_edge_features(u, v)
            X.append(features)
        
        X = np.array(X)
        
        # Get probabilities
        predictions = self.classifier.predict_proba(X)[:, 1]
        
        # Combine and sort
        results = [
            (node_pairs[i][0], node_pairs[i][1], float(predictions[i]))
            for i in range(len(node_pairs))
        ]
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
