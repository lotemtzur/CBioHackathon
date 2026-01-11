# Random Walk Link Prediction - Implementation Summary

## ✅ Implementation Complete

A complete Random Walk-based link prediction system has been implemented with the following components:

### Core Modules

1. **`preprocessing.py`** - Handles isolated nodes by connecting them using preferential attachment based on weighted node strength
2. **`random_walk.py`** - Generates weighted random walks with node2vec-style biased sampling
3. **`embedding_learner.py`** - Learns node embeddings using gensim Word2Vec (Skip-gram)
4. **`rw_prediction.py`** - Binary classifier using Hadamard product features for edge prediction
5. **`evaluate_model.py`** - Complete evaluation pipeline with hyperparameter tuning

### Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run quick example (short dataset)
python RW_baseline/example.py

# Run full evaluation with hyperparameter tuning
python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv

# Run on full dataset (WARNING: computationally intensive!)
python RW_baseline/evaluate_model.py string_interaction_physical.tsv \
    --dimensions 128 \
    --walk-length 80 \
    --num-walks 10 \
    --threshold 0.5 \
    --top-k 100
```

### Example Output

```
Random Walk Link Prediction - Quick Example
============================================================

1. Loading graph...
   Loaded: 175 nodes, 299 edges

4. Generating random walks...
   Generated 1750 walks
   Average walk length: 34.65

5. Learning node embeddings...
   Trained embeddings for 175 nodes
   Embedding shape: (175, 64)

6. Training link predictor...
   Training accuracy: 0.9331

8. Top 10 Predictions:
------------------------------------------------------------
 1. ATG2A <-> WIPI2      | Score: 0.9991 | ✓ IN TEST SET
 2. WIPI1 <-> WIPI2      | Score: 0.9986 | ✗ NOT IN TEST SET
 3. WIPI1 <-> ATG2B      | Score: 0.9977 | ✓ IN TEST SET
 ...
```

### Key Features

✅ **Preprocessing**: Handles isolated nodes automatically  
✅ **Weighted Walks**: Uses edge weights in transition probabilities  
✅ **Node2Vec-style**: Supports p and q parameters for biased walks  
✅ **Word2Vec Embeddings**: Skip-gram learning of node representations  
✅ **Hadamard Features**: Element-wise multiplication for edge features  
✅ **Binary Classifiers**: Logistic Regression or Random Forest  
✅ **Hyperparameter Tuning**: Grid search on validation set  
✅ **Comprehensive Evaluation**: Precision, Recall, F1, Accuracy  

### Architecture

```
Phase 0: Preprocessing
    ↓
Phase 1: Random Walks (walks = sequences of nodes)
    ↓
Phase 2: Embedding Learning (nodes → vectors)
    ↓
Phase 3: Feature Engineering (edge features via Hadamard)
    ↓
Phase 4: Binary Classification (predict edge probability)
```

### Dependencies (Installed in venv)

- networkx >= 2.5
- numpy >= 1.19.0
- pandas >= 1.1.0
- gensim >= 3.8.0 (Word2Vec)
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0

### Next Steps

1. **Run full evaluation on short dataset** to get baseline metrics
2. **Compare with Common Neighbors baseline** (F1 = 0.523)
3. **Tune hyperparameters** if performance is below baseline
4. **Run on full dataset** for final results (may take 30-60 minutes)

### File Structure

```
RW_baseline/
├── __init__.py                  # Package initialization
├── preprocessing.py             # Phase 0: Isolated node handling
├── random_walk.py               # Phase 1: Walk generation
├── embedding_learner.py         # Phase 2: Embedding learning
├── rw_prediction.py             # Phase 3-4: Feature engineering & prediction
├── evaluate_model.py            # Full evaluation pipeline
├── example.py                   # Quick example script
├── README.md                    # Detailed documentation
├── requirements.txt             # Python dependencies
└── output/                      # Results (created after evaluation)
    ├── validation_results.tsv
    ├── test_predictions.tsv
    └── evaluation_results.txt
```

### Performance Tips

- Start with short dataset for faster iteration
- Use `--dimensions 64` for faster training
- Reduce `--num-walks` and `--walk-length` for speed
- Use `--classifier logistic` (faster than Random Forest)
- The full dataset may take 30-60 minutes depending on hyperparameters

### Theoretical Foundation

Based on:
- **DeepWalk** (Perozzi et al., 2014)
- **Node2Vec** (Grover & Leskovec, 2016)
- **Word2Vec** (Mikolov et al., 2013)

The approach captures structural similarity: nodes that co-occur in random walks have similar embeddings, and similar embeddings predict likely edges.
