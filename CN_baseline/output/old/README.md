# Common Neighbors Baseline - Evaluation Results

This folder contains the complete evaluation results for the Common Neighbors baseline model on the full protein-protein interaction dataset.

## Files in This Directory

### Data Files

1. **validation_results.tsv**
   - All 50 hyperparameter combinations tested
   - Columns: threshold, top_k, precision, recall, f1
   - Sorted by F1 score (best configurations first)

2. **test_predictions.tsv**
   - All 922 predictions made on the test set
   - Columns: node1, node2, score, is_correct, actual_in_test
   - Can be used for further analysis of prediction patterns

3. **evaluation_results.txt**
   - Complete console output from the evaluation run
   - Includes dataset statistics, hyperparameter tuning results, and final test performance
   - Full reproducible record of the evaluation

### Visualization Files

1. **metrics_comparison.png**
   - Bar chart comparing Precision, Recall, and F1 Score on test set
   - Shows the balanced performance across all three metrics

2. **hyperparameter_heatmap.png**
   - Heatmap showing F1 scores for different threshold and top-k combinations
   - Helps identify optimal parameter regions
   - Darker colors indicate higher F1 scores

3. **top_configurations.png**
   - Horizontal bar chart of the top 10 hyperparameter configurations
   - Ranked by validation F1 score
   - Shows relative performance of different parameter settings

4. **precision_recall_tradeoff.png**
   - Scatter plot showing precision vs recall for all configurations
   - Color-coded by F1 score
   - Red star marks the best validation configuration
   - Green diamond marks actual test set performance

5. **score_distribution.png**
   - Two-panel figure showing:
     - Left: Histogram of prediction scores (correct vs incorrect)
     - Right: Box plot comparing score distributions
   - Helps understand if higher scores correlate with correctness

6. **evaluation_summary.png**
   - Text summary of key results and interpretation
   - Best hyperparameters and test set performance
   - Quick reference for main findings

## Key Results Summary

### Dataset
- **Total edges**: 4,297 unique protein interactions
- **Total nodes**: 354 proteins
- **Train set**: 3,007 edges (70%)
- **Validation set**: 644 edges (15%)
- **Test set**: 646 edges (15%)

### Best Hyperparameters (Selected on Validation Set)
- **Threshold**: 10.0
- **Top-K**: all (no limit on predictions)
- **Validation F1**: 0.5262

### Test Set Performance
- **Precision**: 0.4447 (44.47%)
- **Recall**: 0.6347 (63.47%)
- **F1 Score**: 0.5230
- **Total Predictions**: 922
- **Correct Predictions**: 410

### Interpretation

The Common Neighbors baseline achieves strong performance:

1. **High Recall (63.47%)**: The model successfully identifies nearly 2/3 of actual protein interactions in the test set

2. **Good Precision (44.47%)**: Almost half of all predictions are correct, indicating the model makes reliable suggestions

3. **Balanced F1 (0.523)**: The model maintains a good balance between precision and recall

4. **Optimal Threshold**: A threshold of 10.0 works best, suggesting that only pairs with very strong common neighbor evidence should be predicted

## Biological Insights

Many top predictions involve mitochondrial proteins (MT-*, NDUF*, COX*, ATP5*, UQCR*), which makes biological sense as these proteins form complex interaction networks in cellular respiration and energy production.

## Using These Results

### For Model Comparison
- Use these metrics as baseline to compare against other methods (Random Walk, GNN)
- Ensure consistent train/val/test split (use same seed=42)

### For Parameter Selection
- Check `validation_results.tsv` for alternative parameter choices
- Different thresholds offer different precision-recall trade-offs

### For Prediction Analysis
- `test_predictions.tsv` contains all predictions with correctness labels
- Can be filtered or analyzed for biological validation
- Note: "Incorrect" predictions may still be biologically valid (not yet discovered)

## Reproducibility

To reproduce these exact results:
```bash
python3 evaluate_model.py ../string_interaction_physical.tsv \
    --threshold-range 1.0 10.0 1.0 \
    --top-k-values 50 100 200 500 \
    --seed 42
```

Random seed 42 ensures the same train/validation/test split every time.
