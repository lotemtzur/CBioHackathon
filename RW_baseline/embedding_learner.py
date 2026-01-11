"""
Embedding learning module using Word2Vec on random walk sequences.

This module implements Phase 2 of the Random Walk link prediction pipeline.
It learns node embeddings by treating random walks as sentences and applying
Skip-gram Word2Vec to capture structural similarity.
"""

import numpy as np
from gensim.models import Word2Vec
from typing import List, Dict, Tuple
import pickle
import os


class EmbeddingLearner:
    """
    Learns node embeddings from random walk sequences using Word2Vec.
    
    Uses Skip-gram model to predict context nodes from a target node,
    learning vector representations that capture structural similarity.
    """
    
    def __init__(self, dimensions: int = 128, window: int = 10, 
                 min_count: int = 0, sg: int = 1, workers: int = 4,
                 epochs: int = 10, seed: int = 42):
        """
        Initialize the embedding learner.
        
        Args:
            dimensions: Size of the embedding vectors (e.g., 64, 128)
            window: Context window size (how many neighbors to consider)
            min_count: Minimum frequency for a node to be included (0 = all nodes)
            sg: Training algorithm (1=skip-gram, 0=CBOW)
            workers: Number of parallel workers
            epochs: Number of training epochs
            seed: Random seed for reproducibility
        """
        self.dimensions = dimensions
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs
        self.seed = seed
        
        self.model = None
        self.node_embeddings = None
    
    def train(self, walks: List[List[str]], verbose: bool = True) -> None:
        """
        Train Word2Vec model on random walk sequences.
        
        Args:
            walks: List of walks, where each walk is a list of node IDs (as strings)
            verbose: Whether to print training progress
        """
        if verbose:
            print(f"\n=== Training Embeddings ===")
            print(f"Number of walks: {len(walks)}")
            print(f"Embedding dimensions: {self.dimensions}")
            print(f"Window size: {self.window}")
            print(f"Epochs: {self.epochs}")
        
        # Convert walks to strings (Word2Vec expects strings)
        walks_str = [[str(node) for node in walk] for walk in walks]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=walks_str,
            vector_size=self.dimensions,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            workers=self.workers,
            epochs=self.epochs,
            seed=self.seed
        )
        
        # Extract embeddings as dictionary
        self.node_embeddings = {
            node: self.model.wv[node] 
            for node in self.model.wv.index_to_key
        }
        
        if verbose:
            print(f"Trained embeddings for {len(self.node_embeddings)} nodes")
            print(f"Embedding shape: ({len(self.node_embeddings)}, {self.dimensions})")
    
    def get_embedding(self, node: str) -> np.ndarray:
        """
        Get embedding vector for a specific node.
        
        Args:
            node: Node ID
            
        Returns:
            Embedding vector (numpy array)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        node_str = str(node)
        if node_str in self.model.wv:
            return self.model.wv[node_str]
        else:
            # Return zero vector for unknown nodes
            return np.zeros(self.dimensions)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get all node embeddings as a dictionary.
        
        Returns:
            Dictionary mapping node ID -> embedding vector
        """
        if self.node_embeddings is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.node_embeddings
    
    def most_similar(self, node: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Find most similar nodes to a given node.
        
        Args:
            node: Node ID
            topn: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        node_str = str(node)
        if node_str in self.model.wv:
            return self.model.wv.most_similar(node_str, topn=topn)
        else:
            return []
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model (should end in .model)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Save Word2Vec model
        self.model.save(filepath)
        
        # Also save embeddings dictionary for faster loading
        embeddings_path = filepath.replace('.model', '_embeddings.pkl')
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.node_embeddings, f)
        
        print(f"Model saved to {filepath}")
        print(f"Embeddings saved to {embeddings_path}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load Word2Vec model
        self.model = Word2Vec.load(filepath)
        
        # Try to load embeddings dictionary
        embeddings_path = filepath.replace('.model', '_embeddings.pkl')
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.node_embeddings = pickle.load(f)
        else:
            # Recreate embeddings from model
            self.node_embeddings = {
                node: self.model.wv[node] 
                for node in self.model.wv.index_to_key
            }
        
        print(f"Model loaded from {filepath}")
        print(f"Loaded embeddings for {len(self.node_embeddings)} nodes")


def learn_embeddings(walks: List[List[str]], 
                    dimensions: int = 128,
                    window: int = 10,
                    epochs: int = 10,
                    seed: int = 42,
                    verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Convenience function to learn embeddings from walks.
    
    Args:
        walks: List of random walks
        dimensions: Embedding vector size
        window: Context window size
        epochs: Training epochs
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Dictionary mapping node ID -> embedding vector
    """
    learner = EmbeddingLearner(
        dimensions=dimensions,
        window=window,
        epochs=epochs,
        seed=seed
    )
    
    learner.train(walks, verbose=verbose)
    
    return learner.get_all_embeddings()
