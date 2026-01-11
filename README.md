# CBioHackathon

## Protein-Protein Interaction Network Analysis and Link Prediction

This project implements and evaluates various methods for predicting protein-protein interactions using graph-based approaches.

## Project Overview

We are developing a comprehensive framework to predict new edges (protein interactions) in biological networks using progressively sophisticated techniques:

### Implemented Methods

1. ✅ **Random Baseline** - Random edge prediction for baseline comparison
2. ✅ **Common Neighbors (CN)** - Link prediction based on shared neighbors
3. 🚧 **Random Walk** - Graph traversal-based prediction (in progress)
4. 🚧 **Graph Neural Networks (GNN)** - Deep learning approach (in progress)

## Dataset

- **Source**: STRING database protein interaction network
- **Files**:
  - `string_interaction_physical.tsv` - Full dataset (~4,300 interactions, 354 proteins)
  - `string_interaction_physical_short.tsv` - Subset for quick testing (~300 interactions, 175 proteins)

## Quick Start

### 1. Common Neighbors Model

#### Basic Prediction
```bash
python3 common_neighbors_prediction.py string_interaction_physical.tsv --threshold 1.0 --top-k 100
```

#### Full Evaluation with Train/Val/Test Split
```bash
python3 evaluate_model.py string_interaction_physical.tsv
```

This will:
- Split data into train (70%), validation (15%), test (15%)
- Tune hyperparameters (threshold, top-k) on validation set
- Evaluate final model on test set
- Generate detailed performance metrics

## Results

### Common Neighbors Baseline

**Full Dataset Performance (4,297 edges, 354 nodes):**
- **Precision**: 44.47%
- **Recall**: 63.47%
- **F1 Score**: 0.523
- **Best Threshold**: 10.0
- **Predictions**: 922 total, 410 correct

The Common Neighbors model provides a strong baseline, correctly predicting nearly half of suggested interactions while finding 2/3 of actual interactions.

## Project Structure

```
CBioHackathon/
├── README.md                           # This file
├── README_common_neighbors.md          # CN model documentation
├── README_evaluation.md                # Evaluation framework guide
├── common_neighbors_prediction.py      # CN implementation
├── evaluate_model.py                   # Train/val/test evaluation
├── string_interaction_physical.tsv     # Full dataset
├── string_interaction_physical_short.tsv # Test dataset
├── validation_results.tsv              # Hyperparameter tuning results
└── test_predictions.tsv                # Final predictions
```

## Documentation

- **[Common Neighbors Model](README_common_neighbors.md)** - Detailed guide for CN baseline
- **[Evaluation Framework](README_evaluation.md)** - How to evaluate models with proper train/val/test splits

## Method Comparison

| Method | Status | Precision | Recall | F1 Score | Notes |
|--------|--------|-----------|--------|----------|-------|
| Random | ✅ | TBD | TBD | TBD | Baseline comparison |
| Common Neighbors | ✅ | 44.47% | 63.47% | 0.523 | Strong baseline |
| Random Walk | 🚧 | - | - | - | In development |
| GNN | 🚧 | - | - | - | In development |

## Requirements

```bash
pip install pandas numpy
```

## Contributing

This is a bioinformatics hackathon project exploring link prediction methods for protein interaction networks.

## Next Steps

1. Implement Random Walk baseline
2. Implement GNN-based approach
3. Compare all methods on consistent train/val/test splits
4. Analyze biological significance of predictions
5. Explore ensemble methods combining multiple approaches
