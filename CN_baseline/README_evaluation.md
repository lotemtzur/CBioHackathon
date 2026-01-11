# Model Evaluation with Train/Validation/Test Split

This guide explains how to use the evaluation script to properly assess the Common Neighbors model performance using train/validation/test splits.

## Overview

The evaluation script (`evaluate_model.py`) implements a rigorous evaluation pipeline:

1. **Data Splitting**: Divides edges into train (70%), validation (15%), and test (15%) sets
2. **Hyperparameter Tuning**: Tests different threshold and top-k combinations on validation set
3. **Best Model Selection**: Chooses parameters that maximize F1 score on validation
4. **Final Evaluation**: Tests the best model on the held-out test set

## Metrics

The script calculates three key metrics:

- **Precision**: Of all predicted edges, what proportion are actually real edges?
  - `Precision = True Positives / (True Positives + False Positives)`
  - Higher is better (fewer false predictions)

- **Recall**: Of all real edges, what proportion did we predict?
  - `Recall = True Positives / Total Actual Edges`
  - Higher is better (finding more real edges)

- **F1 Score**: Harmonic mean of precision and recall
  - `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
  - Balances precision and recall

## Basic Usage

```bash
python3 evaluate_model.py string_interaction_physical_short.tsv
```

This will:
- Split data into 70% train, 15% validation, 15% test
- Test thresholds from 0.5 to 5.0 (step 0.5) with top-k values [10, 20, 50, 100]
- Select best parameters based on validation F1 score
- Evaluate on test set
- Save results to `validation_results.tsv` and `test_predictions.tsv`

## Advanced Options

### Custom Data Split Ratios

```bash
python3 evaluate_model.py INPUT_FILE \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Custom Threshold Range

```bash
python3 evaluate_model.py INPUT_FILE \
    --threshold-range 1.0 10.0 1.0
```

Format: `MIN MAX STEP`

### Custom Top-K Values

```bash
python3 evaluate_model.py INPUT_FILE \
    --top-k-values 10 25 50 100 200
```

### Set Random Seed

```bash
python3 evaluate_model.py INPUT_FILE --seed 123
```

## Complete Example

```bash
python3 evaluate_model.py string_interaction_physical.tsv \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --threshold-range 0.5 5.0 0.5 \
    --top-k-values 10 20 50 100 200 \
    --seed 42
```

## Output Files

### 1. validation_results.tsv

Contains all tested hyperparameter combinations with their validation scores:

```
threshold  top_k  precision  recall     f1
2.5        all    0.166667   0.136364   0.150000
2.0        all    0.162162   0.136364   0.148148
1.5        all    0.134615   0.159091   0.145833
...
```

### 2. test_predictions.tsv

Contains all predictions made on the test set:

```
node1     node2      score   is_correct  actual_in_test
ATP5F1B   ATP5F1D    7.850   True        Yes
ATP5PB    MT-ATP8    6.763   False       No
ATP5MC1   ATP5MC2    6.218   True        Yes
...
```

## Understanding Results

### Example Output

```
============================================================
BEST PARAMETERS (Validation F1=0.1500):
  Threshold: 2.5
  Top-K: all
============================================================

Test Set Results:
  Precision: 0.3056
  Recall: 0.2391
  F1 Score: 0.2683
  Total predictions: 36
  Correct predictions: 11
```

**Interpretation:**
- Best threshold found: 2.5 (common neighbor score minimum)
- Using all predictions above threshold (no top-k limit)
- Test precision of 30.56% means ~1 in 3 predictions is correct
- Test recall of 23.91% means we found ~1 in 4 real edges
- F1 score balances these at 0.2683

### Top Predictions

```
Top 10 predictions on test set:
  1. ATP5F1B <-> ATP5F1D: 7.850 [✓ CORRECT]
  2. ATP5PB <-> MT-ATP8: 6.763 [✗ INCORRECT]
  3. ATP5MC1 <-> ATP5MC2: 6.218 [✓ CORRECT]
```

- ✓ CORRECT: Prediction matches actual edge in test set
- ✗ INCORRECT: Prediction doesn't exist in test set (but might still be biologically valid!)

## Key Considerations

### 1. Train/Test Contamination
The script ensures **no overlap** between train/validation/test sets:
- Training: Used to build the model (adjacency list for common neighbors)
- Validation: Used to select hyperparameters (never used for training)
- Test: Used only once for final evaluation (never used for training or tuning)

### 2. Undirected Graph Handling
The script properly handles edges as undirected:
- Each edge is stored in both directions during training
- Predictions check both directions for correctness
- Prevents counting the same edge twice

### 3. Random Seed
Use `--seed` for reproducibility:
- Same seed = same train/val/test split every time
- Important for comparing different models fairly

### 4. Hyperparameter Space
More combinations = better optimization but slower:
- Threshold range: Controls minimum score for predictions
- Top-k values: Controls how many predictions to make
- Consider your computational budget vs. optimization quality

## Best Practices

1. **Use consistent splits**: Same train/val/test for comparing different models
2. **Never tune on test**: Only look at test results after selecting final parameters
3. **Multiple seeds**: Run with different seeds to assess stability
4. **Appropriate metrics**: Choose metrics based on your use case:
   - High precision needed? → Optimize for precision (accept lower recall)
   - Find all edges? → Optimize for recall (accept lower precision)
   - Balanced? → Optimize for F1 score

## Example Workflow

```bash
# 1. Quick test on small dataset
python3 evaluate_model.py string_interaction_physical_short.tsv

# 2. Full evaluation on complete dataset with custom parameters
python3 evaluate_model.py string_interaction_physical.tsv \
    --threshold-range 1.0 10.0 1.0 \
    --top-k-values 50 100 200 500

# 3. Check validation results
cat validation_results.tsv | head -20

# 4. Analyze test predictions
cat test_predictions.tsv | grep "Yes" | wc -l  # Count correct predictions
```

## Performance Notes

- **Short dataset (~300 edges)**: Takes ~1-2 seconds
- **Full dataset (~8,600 edges)**: Takes ~30-60 seconds depending on hyperparameter space
- Time scales with: number of nodes² × number of hyperparameter combinations

## Troubleshooting

**Q: My test F1 is much lower than validation F1**
- A: This is normal! It indicates the model may be slightly overfit to validation set
- Solution: Try simpler parameter ranges or use cross-validation

**Q: All predictions are incorrect (precision = 0)**
- A: Threshold might be too low or training data too limited
- Solution: Try lower thresholds or increase training data ratio

**Q: Recall is very low**
- A: Not finding many real edges
- Solution: Lower threshold or increase top-k to make more predictions

**Q: Script is too slow**
- A: Too many hyperparameter combinations
- Solution: Reduce threshold range step size or reduce top-k values
