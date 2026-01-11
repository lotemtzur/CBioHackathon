# Random Walk Link Prediction

This directory contains a Random Walk-based link prediction implementation using node embeddings and binary classification.

## Overview

The approach follows a multi-phase pipeline inspired by Node2Vec and DeepWalk:

### Phase 0: Preprocessing
- **Goal**: Ensure all nodes are connected for random walk traversal
- **Method**: Calculate weighted node strength $S_i = \sum_j Weight_{ij}$ and attachment probabilities $P_i = S_i / \sum S$
- **Action**: Connect isolated nodes with synthetic edges (weight = graph average)

### Phase 1: Weighted Random Walks
- **Goal**: Generate node sequences that capture graph structure
- **Method**: Biased random walks with transition probabilities weighted by edge weights
- **Output**: List of "sentences" (node sequences) representing structural context

### Phase 2: Embedding Learning
- **Goal**: Convert walks into dense vector representations
- **Method**: Word2Vec Skip-gram on walk sequences (similar to Node2Vec)
- **Output**: Node embeddings (e.g., 64 or 128-dimensional vectors)

### Phase 3: Dataset Construction
- **Goal**: Prepare training data for binary classifier
- **Method**: 
  - Positive samples: Existing edges
  - Negative samples: Non-existing edges (balanced sampling)
  - Features: Hadamard product (element-wise multiplication) of node embeddings
- **Formula**: $Feature_{edge} = Vector_u \odot Vector_v$

### Phase 4: Training & Inference
- **Goal**: Predict edge probabilities
- **Method**: Train Logistic Regression or Random Forest on edge features
- **Output**: Probability scores for all candidate edges (sorted by confidence)

## Files

- **`preprocessing.py`**: Handles isolated nodes by adding synthetic edges
- **`random_walk.py`**: Generates weighted random walks with node2vec-style biased sampling
- **`embedding_learner.py`**: Learns node embeddings using gensim Word2Vec
- **`rw_prediction.py`**: Binary classifier for link prediction using Hadamard features
- **`evaluate_model.py`**: Complete evaluation pipeline with hyperparameter tuning

## Usage

### Basic Usage

```bash
# Run evaluation with default parameters
python RW_baseline/evaluate_model.py string_interaction_physical.tsv
```

### Advanced Usage

```bash
# Custom hyperparameter search
python RW_baseline/evaluate_model.py string_interaction_physical.tsv \
    --dimensions 64 128 \
    --walk-length 20 40 80 \
    --num-walks 5 10 20 \
    --threshold 0.3 0.5 0.7 \
    --top-k 50 100 200 \
    --classifier logistic \
    --seed 42
```

### Parameters

#### Data Splitting
- `--train-ratio`: Training set ratio (default: 0.7)
- `--val-ratio`: Validation set ratio (default: 0.15)
- `--test-ratio`: Test set ratio (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

#### Random Walk Hyperparameters
- `--dimensions`: Embedding dimensions to test (default: [64, 128])
- `--walk-length`: Walk lengths to test (default: [20, 40, 80])
- `--num-walks`: Number of walks per node to test (default: [5, 10])

#### Prediction Hyperparameters
- `--threshold`: Prediction thresholds to test (default: [0.3, 0.5, 0.7])
- `--top-k`: Top-K values to test (default: [50, 100, 200])
- `--classifier`: Classifier type: 'logistic' or 'rf' (default: logistic)

## Output

Results are saved to `RW_baseline/output/`:

- **`validation_results.tsv`**: All hyperparameter combinations with metrics
- **`test_predictions.tsv`**: Final predictions on test set
- **`evaluation_results.txt`**: Summary of best configuration and test performance

## Evaluation Metrics

- **Precision**: Percentage of predicted edges that are correct
- **Recall**: Percentage of actual test edges found
- **F1 Score**: Harmonic mean of precision and recall (optimization target)
- **Accuracy**: Percentage of all predictions (positive + negative) that are correct

## Algorithm Details

### Random Walk Generation

The random walker uses weighted transition probabilities:

$$P(v_k | u) = \frac{Weight(u, v_k)}{\sum_{neighbors} Weight(u, neighbor)}$$

For node2vec-style biased walks with parameters $p$ and $q$:
- If $v$ is the previous node: weight / $p$
- If $v$ is neighbor of previous: weight (unchanged)
- Otherwise: weight / $q$

(Default: $p=1$, $q=1$ for unbiased walks)

### Embedding Learning

Uses gensim Word2Vec with:
- **Algorithm**: Skip-gram (predicts context from target)
- **Window size**: 10 (context window for co-occurrence)
- **Epochs**: 10 (training iterations)
- **Min count**: 0 (include all nodes)

### Link Prediction

For each candidate edge $(u, v)$:
1. Get embeddings: $\vec{u}$ and $\vec{v}$
2. Compute Hadamard product: $\vec{f} = \vec{u} \odot \vec{v}$
3. Feed $\vec{f}$ into classifier
4. Output probability: $P(edge_{u,v} = 1)$

## Dependencies

Required Python packages:
```bash
pip install networkx numpy pandas gensim scikit-learn matplotlib seaborn
```

## Example Output

```
=============================================================
BEST PARAMETERS (Validation F1=0.5847):
  Dimensions: 128
  Walk Length: 80
  Num Walks: 10
  Threshold: 0.5
  Top-K: 100
=============================================================

FINAL EVALUATION ON TEST SET
=============================================================

Test Set Results:
  Accuracy: 0.9234
  Precision: 0.5200
  Recall: 0.6500
  F1 Score: 0.5775
  Total predictions: 100
  True positives: 65
```

## Comparison with Baselines

| Model | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| Random Baseline | ~0.01 | ~0.50 | ~0.02 |
| Common Neighbors | 0.4447 | 0.6347 | **0.5230** |
| **Random Walk (This)** | **TBD** | **TBD** | **TBD** |

*Target: Beat Common Neighbors F1 = 0.523*

## Theory

### Why Random Walks Work

1. **Structural Similarity**: Nodes that appear together in random walks share similar graph neighborhoods
2. **Weighted Context**: Edge weights influence walk transitions, capturing importance
3. **Embedding Space**: Word2Vec learns that nodes with similar contexts get similar vectors
4. **Feature Combination**: Hadamard product captures interaction between node embeddings

### Advantages
- Captures higher-order proximity (beyond common neighbors)
- Handles weighted graphs naturally
- Learns distributed representations
- Scalable with parallel walks

### Limitations
- Computationally expensive (walks + embedding training)
- Many hyperparameters to tune
- Requires balanced negative sampling
- May overfit on small graphs

## References

- **Node2Vec**: Grover & Leskovec (2016) - "node2vec: Scalable Feature Learning for Networks"
- **DeepWalk**: Perozzi et al. (2014) - "DeepWalk: Online Learning of Social Representations"
- **Word2Vec**: Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space"

## Future Improvements

1. Try concatenation or other edge feature methods
2. Experiment with $p$ and $q$ parameters for biased walks
3. Use deeper classifiers (MLP, XGBoost)
4. Implement attention-based walk sampling
5. Add graph-level features (clustering coefficient, centrality)
