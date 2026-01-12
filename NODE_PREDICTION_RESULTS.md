# Node Prediction Results

## Task: Predict Interactions for Isolated Nodes

### Problem Setup
- **Challenge**: Predict interaction partners for completely isolated test nodes (20% of proteins)
- **Input**: Only protein sequence (ESM embeddings) for test nodes
- **Approach**: Adamic-Adar algorithm enhanced with sequence features

### Results

#### Adamic-Adar + Sequence Features

**Final Performance (Test Set, 70 nodes):**
- **Precision: 51.0%** ← About half of top-10 predictions are correct
- **Recall: 19.3%** ← Finds ~1/5 of actual interactions (with top-10 limit)
- **F1 Score: 0.2798**
- **AUC: 0.7590** ← Strong discriminative ability across all thresholds

**Hyperparameters:**
- **k = 5** (number of virtual neighbors based on sequence similarity)
- **Top-k predictions = 10** per test node

**Algorithm Overview:**
1. Find k=5 most similar training nodes by ESM sequence similarity
2. These become "virtual neighbors" connecting test node to the network
3. Score potential partners using Adamic-Adar through virtual neighbors
4. Formula: `score(test, partner) = Σ seq_sim(test, v_i) × (1/log(degree(v_i)))`

### Sample Predictions

#### Perfect Predictions (10/10 correct)
- **COX4I1** (mitochondrial protein): All predicted partners correct
- **WNT1** (Wnt signaling): All predicted FZD receptors and LRP5 correct

#### Good Predictions (4-5/10 correct)  
- **NOS2**: 4/10 correct (CALML proteins identified)
- **AGER**: 2/10 correct (LRP1, MAPK1)

#### Challenging Cases
- **CSF1**: 0/10 correct (only 3 actual neighbors, rare interactions)

### Comparison Context

This is a **node-level (inductive) prediction task**, which is more challenging than edge prediction:

**Edge Prediction (existing baselines):**
- Common Neighbors: 44.5% precision, 63.5% recall
- Random Walk: ~similar performance on held-out edges

**Node Prediction (this work):**
- Adamic-Adar + Sequences: 51.0% precision, 19.3% recall
- Challenge: Test nodes are completely isolated (no edges during training)

### Key Insights

1. **Sequence similarity works**: ESM embeddings successfully bridge isolated nodes to the network
2. **Lower k is better**: k=5 outperforms k=30, suggesting focused neighbors are more informative
3. **Precision > Recall**: High precision (51%) shows predictions are reliable, low recall due to top-10 limit
4. **Strong AUC (0.7590)**: Model has good discriminative ability across all score thresholds
5. **Biological coherence**: Works best for protein families (COX*, WNT*, FZD*)

### Files

- **Implementation**: `adamic_adar_sequence.py`
- **Utilities**: `utils.py` (added embedding loading and cosine similarity)
- **Pre-computed embeddings**: `protein_embeddings.pkl` (ESM-2 650M)
- **Hyperparameter tuning plot**: `adamic_adar_k_tuning.png`
- **ROC curve**: `adamic_adar_node_roc.png` (AUC = 0.7590)

### Next Steps

1. **Compare to FFN baseline**: Train the logistic regression baseline you mentioned
2. **Try other k values**: Fine-tune on specific protein families
3. **Ensemble approach**: Combine with network-only methods
4. **Evaluate by degree**: Test performance for hub vs. sparse proteins
