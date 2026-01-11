"""
Random Walk Link Prediction Package

A complete implementation of link prediction using:
- Weighted random walks for graph exploration
- Node2Vec-style embeddings via Word2Vec
- Hadamard product features for edge representation
- Binary classification (Logistic Regression / Random Forest)
"""

from .preprocessing import preprocess_graph, connect_isolated_nodes
from .random_walk import generate_random_walks, WeightedRandomWalker
from .embedding_learner import EmbeddingLearner, learn_embeddings
from .rw_prediction import RWPredictor

__all__ = [
    'preprocess_graph',
    'connect_isolated_nodes',
    'generate_random_walks',
    'WeightedRandomWalker',
    'EmbeddingLearner',
    'learn_embeddings',
    'RWPredictor'
]

__version__ = '1.0.0'
