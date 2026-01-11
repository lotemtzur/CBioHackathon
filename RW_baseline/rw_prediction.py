"""
Random Walk link prediction using node embeddings and binary classification.

This module implements Phases 3-4 of the Random Walk link prediction pipeline:
- Phase 3: Dataset construction with Hadamard product features
- Phase 4: Training and inference using binary classifiers
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random


class RWPredictor:
    """
    Link prediction using Random Walk embeddings and binary classification.
    
    Combines node embeddings using Hadamard product (element-wise multiplication)
    and trains a classifier to predict edge existence.
    """
    
    def __init__(self, embeddings: Dict[str, np.ndarray], 
                 classifier_type: str = 'logistic',
                 random_state: int = 42):
        """
        Initialize the predictor.
        
        Args:
            embeddings: Dictionary mapping node ID -> embedding vector
            classifier_type: 'logistic' for Logistic Regression, 'rf' for Random Forest
            random_state: Random seed for reproducibility
        """
        self.embeddings = embeddings
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.classifier = None
        
        # Get embedding dimension
        first_embedding = next(iter(embeddings.values()))
        self.embedding_dim = len(first_embedding)
    
    def _compute_edge_feature(self, node1: str, node2: str, 
                             method: str = 'hadamard') -> np.ndarray:
        """
        Compute edge feature vector by combining node embeddings.
        
        Args:
            node1: First node ID
            node2: Second node ID
            method: Combination method ('hadamard', 'average', 'concat')
            
        Returns:
            Feature vector for the edge
        """
        # Get embeddings (use zero vector if node not found)
        emb1 = self.embeddings.get(str(node1), np.zeros(self.embedding_dim))
        emb2 = self.embeddings.get(str(node2), np.zeros(self.embedding_dim))
        
        if method == 'hadamard':
            # Element-wise multiplication (recommended)
            return emb1 * emb2
        elif method == 'average':
            # Average of the two vectors
            return (emb1 + emb2) / 2
        elif method == 'concat':
            # Concatenation
            return np.concatenate([emb1, emb2])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_negative_samples(self, G: nx.Graph, num_samples: int,
                                   existing_edges: set) -> List[Tuple[str, str]]:
        """
        Generate negative samples (non-existing edges) for training.
        
        Args:
            G: Graph with training edges
            num_samples: Number of negative samples to generate
            existing_edges: Set of existing edges (to avoid sampling them)
            
        Returns:
            List of (node1, node2) tuples for negative samples
        """
        nodes = list(G.nodes())
        negative_samples = []
        
        max_attempts = num_samples * 10  # Avoid infinite loop
        attempts = 0
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            # Randomly sample two different nodes
            node1, node2 = random.sample(nodes, 2)
            
            # Create canonical edge representation (sorted tuple)
            edge = tuple(sorted([node1, node2]))
            
            # Check if edge doesn't exist
            if edge not in existing_edges and not G.has_edge(node1, node2):
                negative_samples.append(edge)
            
            attempts += 1
        
        if len(negative_samples) < num_samples:
            print(f"Warning: Could only generate {len(negative_samples)} negative samples "
                  f"out of {num_samples} requested")
        
        return negative_samples
    
    def train(self, G_train: nx.Graph, method: str = 'hadamard',
              balance_ratio: float = 1.0, verbose: bool = True) -> None:
        """
        Train the binary classifier on the training graph.
        
        Args:
            G_train: Training graph (edges = positive samples)
            method: Feature combination method
            balance_ratio: Ratio of negative to positive samples
            verbose: Print training progress
        """
        if verbose:
            print("\n=== Training Link Predictor ===")
            print(f"Classifier type: {self.classifier_type}")
            print(f"Feature method: {method}")
        
        # Generate positive samples from existing edges
        positive_edges = list(G_train.edges())
        num_positive = len(positive_edges)
        
        if verbose:
            print(f"Positive samples: {num_positive}")
        
        # Create set of existing edges (canonical form)
        existing_edges = {tuple(sorted([u, v])) for u, v in positive_edges}
        
        # Generate negative samples
        num_negative = int(num_positive * balance_ratio)
        negative_edges = self._generate_negative_samples(
            G_train, num_negative, existing_edges
        )
        
        if verbose:
            print(f"Negative samples: {len(negative_edges)}")
        
        # Prepare training data
        X = []
        y = []
        
        # Add positive samples (label = 1)
        for u, v in positive_edges:
            feature = self._compute_edge_feature(u, v, method)
            X.append(feature)
            y.append(1)
        
        # Add negative samples (label = 0)
        for u, v in negative_edges:
            feature = self._compute_edge_feature(u, v, method)
            X.append(feature)
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        if verbose:
            print(f"Training data shape: {X.shape}")
            print(f"Positive class: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
            print(f"Negative class: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
        
        # Train classifier
        if self.classifier_type == 'logistic':
            self.classifier = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.classifier_type == 'rf':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        self.classifier.fit(X, y)
        
        if verbose:
            train_score = self.classifier.score(X, y)
            print(f"Training accuracy: {train_score:.4f}")
    
    def predict(self, node_pairs: List[Tuple[str, str]], 
                method: str = 'hadamard',
                return_proba: bool = True) -> List[Tuple[str, str, float]]:
        """
        Predict edge probabilities for given node pairs.
        
        Args:
            node_pairs: List of (node1, node2) tuples to predict
            method: Feature combination method (should match training)
            return_proba: Return probabilities (True) or binary predictions (False)
            
        Returns:
            List of (node1, node2, score) tuples sorted by score (descending)
        """
        if self.classifier is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Compute features for all pairs
        X = []
        for u, v in node_pairs:
            feature = self._compute_edge_feature(u, v, method)
            X.append(feature)
        
        X = np.array(X)
        
        # Get predictions
        if return_proba:
            # Return probability of positive class
            predictions = self.classifier.predict_proba(X)[:, 1]
        else:
            # Return binary predictions
            predictions = self.classifier.predict(X)
        
        # Combine with node pairs and sort by score
        results = [
            (node_pairs[i][0], node_pairs[i][1], float(predictions[i]))
            for i in range(len(node_pairs))
        ]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def predict_new_edges(self, G: nx.Graph, top_k: int = None,
                         threshold: float = 0.5,
                         method: str = 'hadamard',
                         verbose: bool = True) -> List[Tuple[str, str, float]]:
        """
        Predict new edges for non-connected node pairs in the graph.
        
        Args:
            G: Current graph (edges to exclude from predictions)
            top_k: Return only top-k predictions (None = all)
            threshold: Minimum probability threshold for predictions
            method: Feature combination method
            verbose: Print progress
            
        Returns:
            List of (node1, node2, probability) tuples for predicted edges
        """
        if verbose:
            print("\n=== Predicting New Edges ===")
        
        nodes = list(G.nodes())
        
        # Generate all possible node pairs (excluding existing edges)
        candidate_pairs = []
        existing_edges = {tuple(sorted([u, v])) for u, v in G.edges()}
        
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                edge = tuple(sorted([u, v]))
                if edge not in existing_edges:
                    candidate_pairs.append((u, v))
        
        if verbose:
            print(f"Candidate pairs: {len(candidate_pairs)}")
        
        # Predict probabilities
        predictions = self.predict(candidate_pairs, method=method, return_proba=True)
        
        # Filter by threshold
        filtered_predictions = [
            (u, v, score) for u, v, score in predictions if score >= threshold
        ]
        
        if verbose:
            print(f"Predictions above threshold ({threshold}): {len(filtered_predictions)}")
        
        # Return top-k if specified
        if top_k is not None:
            filtered_predictions = filtered_predictions[:top_k]
            if verbose:
                print(f"Returning top-{top_k} predictions")
        
        return filtered_predictions
