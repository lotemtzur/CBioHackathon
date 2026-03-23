# CBioHackathon

Protein-Protein Interaction (PPI) link prediction on the STRING database network. We benchmark classical graph heuristics against GNN-based methods across two task types: **edge prediction** (transductive) and **node prediction** (inductive / cold-start).

## Dataset

| File | Description |
|------|-------------|
| `string_interaction_physical.tsv` | Full STRING network (~4 300 edges, 354 proteins) |
| `string_interactions_short.tsv` | Subset for fast iteration (~300 edges, 175 proteins) |
| `string_protein_sequences.fa` | FASTA protein sequences (for ESM-2 embeddings) |
| `protein_embeddings.pkl` | Pre-computed ESM-2 (650M) embeddings |

## Methods

### Edge Prediction (Transductive)

| Method | Script | Key Idea |
|--------|--------|----------|
| Random Baseline | `random_baseline.py` | Uniform random scores -- lower bound |
| Common Neighbors | `CN_baseline/` | Weighted CN scoring with threshold tuning |
| Random Walk w/ Restart | `markov_baseline.py` | RWR affinity matrix with alpha tuning |
| Sequence-only MLP | `pred_by_seq_baseline.py` | Concatenated ESM-2 pair embeddings, MLP classifier (PyTorch Lightning) |

### Node Prediction (Inductive / Cold-Start)

| Method | Script | Key Idea |
|--------|--------|----------|
| Adamic-Adar + Sequence | `adamic_adar_sequence.py` | Sequence similarity creates virtual neighbors; score via AA index |
| VGAE + MLP (GCN) | `gnn_lightning_metrics.py` | VGAE learns structure; MLP maps ESM to latent for unseen nodes |
| VGAE + MLP (GAT) | `ALON_BEST_gat_lightning_metrics.py` | Same pipeline with multi-head GAT encoder |
| Node-deletion GNN | `node_deletion_gnn.py` | VGAE on partial graph; evaluate reconstruction of deleted nodes |

### Supporting Code

| File | Purpose |
|------|---------|
| `utils.py` | Graph loading, train/val/test splits (edge, vertex, semi-inductive), metrics, ROC plotting |
| `extract_proteins_representations.py` | Extract ESM-2 embeddings from FASTA to pickle |
| `alon_files/` | VGAE experiments, topology analysis, CN vs GNN comparison scripts |

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate protein embeddings (one-time, ~10 min)
python extract_proteins_representations.py --fasta string_protein_sequences.fa

# Run baselines
python random_baseline.py
python markov_baseline.py

# Run sequence-based classifier
python pred_by_seq_baseline.py

# Run cold-start node prediction
python adamic_adar_sequence.py
python gnn_lightning_metrics.py
```

## Requirements

```
torch>=2.0.0
fair-esm>=2.0.0
torch-geometric
pytorch-lightning
torchmetrics
scikit-learn
numpy>=1.24.0
pandas>=2.0.0
networkx>=3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Project Structure

```
CBioHackathon/
├── utils.py                          # Shared graph utilities, splits and metrics
├── extract_proteins_representations.py # ESM-2 embedding extraction
├── random_baseline.py                # Random baseline
├── markov_baseline.py                # Random Walk with Restart
├── pred_by_seq_baseline.py           # Sequence-only MLP (Lightning)
├── adamic_adar_sequence.py           # AA + sequence for node prediction
├── gnn_lightning_metrics.py          # VGAE cold-start (GCN encoder)
├── ALON_BEST_gat_lightning_metrics.py # VGAE cold-start (GAT encoder)
├── ALON_BEST_gnn_lightning_metrics.py # VGAE cold-start (GCN, with t-SNE viz)
├── node_deletion_gnn.py              # Inductive node-deletion experiment
├── CN_baseline/                      # Common Neighbors baseline and evaluation
├── alon_files/                       # VGAE experiments and topology analysis
├── Report/                           # Report guidelines
├── string_interaction_physical.tsv   # Full dataset
├── string_interactions_short.tsv     # Small dataset
├── string_protein_sequences.fa       # Protein sequences
├── protein_embeddings.pkl            # Pre-computed ESM-2 embeddings
└── requirements.txt
```
