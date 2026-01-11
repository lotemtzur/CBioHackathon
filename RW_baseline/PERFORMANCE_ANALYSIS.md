# Performance Analysis & Improvements

## Current Results (Poor)

**Test Set Performance:**
- **F1 Score: 0.137** (Target: 0.523 from Common Neighbors baseline)
- **Precision: 0.100** (only 10% of predictions correct)
- **Recall: 0.217** (finding 22% of actual edges)
- **Status: ❌ UNDERPERFORMING** by 73.8%

## Why It's Failing

### 1. **Hyperparameters Too Conservative**
- Walk length = 20 (too short to capture context)
- Num walks = 5 (insufficient sampling)
- Result: Embeddings don't capture enough structural information

### 2. **Sparse Graph Challenge**
- Protein interaction networks are inherently sparse
- Many edges connect nodes with NO common neighbors
- Pure random walk approaches struggle with sparsity

### 3. **Feature Limitations**
- Hadamard product alone may not capture all edge patterns
- Missing topological features (degree, clustering, etc.)
- No hybrid approach combining embeddings with graph metrics

## Solutions Implemented

### ✅ Solution 1: Better Hyperparameters

**Updated defaults in `evaluate_model.py`:**
```python
# OLD (Poor)
dimensions = [64, 128]      # → NEW: [128]
walk_length = [20, 40, 80]  # → NEW: [40, 80]
num_walks = [5, 10]         # → NEW: [10, 20]
threshold = [0.3, 0.5, 0.7] # → NEW: [0.5, 0.6, 0.7]
```

**Rationale:**
- **Longer walks** (40-80): Capture more distant relationships
- **More walks** (10-20): Better sampling of graph structure
- **Higher thresholds** (0.5-0.7): More conservative, fewer false positives
- **Focus on 128 dims**: Better embeddings with reasonable computation

### ✅ Solution 2: Hybrid Predictor

**New file: `hybrid_predictor.py`**

Combines multiple approaches:
1. **Random Walk embeddings** (Hadamard product) - captures global structure
2. **Common Neighbors** - captures local topology
3. **Adamic-Adar** - weighted common neighbors
4. **Degree features** - preferential attachment
5. **Jaccard coefficient** - neighborhood overlap

**Feature vector (133 dimensions):**
```
[embedding_hadamard_64 dims, CN_count, Adamic_Adar, degree_sum, pref_attach, jaccard]
```

### ✅ Solution 3: Diagnostic Tools

**New file: `diagnose.py`**

Analyzes:
- Embedding quality (variance, similarity)
- Prediction score distributions
- Comparison with Common Neighbors baseline
- Identifies specific failure modes

## How to Use Improvements

### Option 1: Run with Better Hyperparameters (Recommended)

```bash
source venv/bin/activate

# Updated defaults should perform better
python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv
```

### Option 2: Run Diagnostics First

```bash
# Analyze what's wrong
python RW_baseline/diagnose.py
```

This will show:
- Embedding quality metrics
- Prediction score distributions
- Common neighbor statistics
- Specific recommendations

### Option 3: Use Hybrid Predictor (Best Performance Expected)

Create a new evaluation script using `HybridRWPredictor` instead of `RWPredictor`. The hybrid approach should significantly outperform pure Random Walk.

### Option 4: Manual Hyperparameter Tuning

```bash
# Aggressive: More computation, potentially better results
python RW_baseline/evaluate_model.py string_interaction_physical_short.tsv \
    --dimensions 128 \
    --walk-length 80 \
    --num-walks 20 \
    --threshold 0.6 0.7 \
    --top-k 100 \
    --classifier rf  # Try Random Forest instead of Logistic Regression
```

## Expected Improvements

| Approach | Expected F1 | Reasoning |
|----------|-------------|-----------|
| **Original** | 0.137 | Too conservative, insufficient context |
| **Better Hyperparams** | 0.25-0.35 | Longer walks, more samples |
| **Hybrid Model** | 0.40-0.55 | Combines embeddings + topology |
| **Target (CN Baseline)** | 0.523 | Goal to beat |

## Alternative Approaches to Consider

### 1. **Graph Neural Networks (GNN)**
- More sophisticated than random walks
- Libraries: PyTorch Geometric, DGL
- Best for: Large graphs, when computational resources allow

### 2. **Matrix Factorization**
- SVD on adjacency matrix
- Simpler and often surprisingly effective
- Best for: When interpretability matters

### 3. **Ensemble Methods**
- Combine CN + Random Walk + Matrix Factorization
- Weighted voting or meta-learning
- Best for: Maximum performance, competition settings

### 4. **Feature Engineering**
- Add more hand-crafted features:
  - Shortest path length
  - Clustering coefficient
  - PageRank scores
  - Katz centrality
- Best for: Domain knowledge integration

## Key Takeaway

**The fundamental issue**: Random Walk alone struggles with sparse graphs. The protein interaction network doesn't have enough "paths" for random walks to reliably capture all structural patterns.

**The solution**: Hybrid approaches that combine:
- **Global structure** (embeddings) 
- **Local topology** (common neighbors, degrees)
- **Better hyperparameters** (more walks, longer walks)

This is why Common Neighbors performs relatively well - it directly measures local connectivity, which matters more in sparse biological networks!
