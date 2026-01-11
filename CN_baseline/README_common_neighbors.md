# Common Neighbors Edge Prediction Model

This Python script implements a baseline Common Neighbors model for predicting new edges in protein-protein interaction networks.

## Algorithm Overview

For every pair of nodes (x, y) that don't have an edge between them, the algorithm calculates a score based on their common neighbors:

```
Score(x, y) = Σ min(combined_score(z, x), combined_score(z, y))
              for all common neighbors z
```

The intuition is that if two proteins share many neighbors with strong interaction scores, they are likely to interact themselves.

## Requirements

- Python 3.x
- pandas
- numpy

Install dependencies:
```bash
pip install pandas numpy
```

## Usage

### Basic Usage

```bash
python3 common_neighbors_prediction.py string_interaction_physical_short.tsv
```

This will:
- Load the protein interaction network
- Calculate common neighbor scores for all non-connected node pairs
- Predict edges with scores above threshold 0.5 (default)
- Save predictions to `predicted_edges.tsv`

### Advanced Options

```bash
python3 common_neighbors_prediction.py INPUT_FILE [OPTIONS]
```

**Options:**
- `--threshold FLOAT`: Minimum common neighbor score threshold (default: 0.5)
- `--top-k INT`: Return only top K predictions (default: all above threshold)
- `--output FILE`: Output file path (default: predicted_edges.tsv)

### Examples

**Get top 20 predictions with threshold of 1.0:**
```bash
python3 common_neighbors_prediction.py string_interaction_physical_short.tsv \
    --threshold 1.0 --top-k 20 --output top20_predictions.tsv
```

**Get all predictions above threshold 2.0:**
```bash
python3 common_neighbors_prediction.py string_interaction_physical_short.tsv \
    --threshold 2.0 --output high_confidence_predictions.tsv
```

**Run on full dataset:**
```bash
python3 common_neighbors_prediction.py string_interaction_physical.tsv \
    --threshold 5.0 --top-k 100
```

## Input Format

The input TSV file should have the following format:
- First line starts with `#` and contains column headers
- Required columns: `node1`, `node2`, `combined_score`
- Tab-separated values

Example:
```
#node1	node2	node1_string_id	node2_string_id	...	combined_score
ADAM10	PPIF	9606.ENSP00000260408	9606.ENSP00000225174	...	0.720
ADAM10	APH1A	9606.ENSP00000260408	9606.ENSP00000358105	...	0.505
```

## Output Format

The output TSV file contains three columns:
- `node1`: First protein in predicted interaction
- `node2`: Second protein in predicted interaction
- `common_neighbor_score`: Calculated score based on common neighbors

Example output:
```
node1	node2	common_neighbor_score
ATP5PB	MT-ATP8	7.66
ATP5PB	ATP5PO	7.631
ATP5PO	MT-ATP8	7.583
```

Higher scores indicate stronger evidence for the predicted interaction based on shared neighbors.

## How It Works

1. **Load Graph**: Reads the protein interaction network from TSV file
2. **Build Adjacency List**: Creates efficient data structure for neighbor lookup
3. **Score Calculation**: For each non-connected pair (x, y):
   - Find common neighbors (nodes connected to both x and y)
   - For each common neighbor z, take min(score(z,x), score(z,y))
   - Sum these minimum scores
4. **Filtering**: Apply threshold and/or top-K filtering
5. **Output**: Save predictions ranked by score

## Performance Notes

- For a graph with N nodes, the algorithm evaluates O(N²) node pairs
- Processing time scales with the number of nodes and average degree
- For the short dataset (175 nodes), processing takes a few seconds
- For larger datasets, consider using a higher threshold or top-k to limit output

## Example Run

```bash
$ python3 common_neighbors_prediction.py string_interaction_physical_short.tsv --threshold 1.0 --top-k 20

Loading graph from string_interaction_physical_short.tsv...

Predicting edges with Common Neighbors model...
Threshold: 1.0
Top K: 20
Total nodes: 175
Existing edges: 299
Processing node 0/175...
Processing node 100/175...
Total node pairs evaluated: 15225
Pairs with common neighbors: 2007
Predictions above threshold 1.0: 117

=== Top 10 Predictions ===
ATP5PB <-> MT-ATP8: 7.660
ATP5PB <-> ATP5PO: 7.631
ATP5PO <-> MT-ATP8: 7.583
ATP5PB <-> ATP5PD: 6.872
MT-ATP6 <-> ATP5PD: 6.871
ATP5PB <-> MT-ATP6: 6.868
MT-ATP6 <-> ATP5PF: 6.844
ATP5PD <-> ATP5PF: 6.841
ATP5PB <-> ATP5PF: 6.838
MT-ATP6 <-> MT-ATP8: 6.762

Predictions saved to predicted_edges.tsv

=== Statistics ===
Total predictions: 20
Score range: [2.959, 7.660]
Mean score: 6.549
Median score: 6.762
