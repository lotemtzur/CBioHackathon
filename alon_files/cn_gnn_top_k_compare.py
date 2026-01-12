import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.nn import VGAE, GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_dense_adj


# ==========================================
# 1. Model Definitions
# ==========================================

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# ==========================================
# 2. Data Loading (One-Hot Features)
# ==========================================

def load_data(file_path):
    print(f"--- Loading data from {file_path} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    df = pd.read_csv(file_path, sep='\t')
    df.rename(columns={col: col.lstrip('#').strip() for col in df.columns}, inplace=True)

    col1 = next((c for c in df.columns if c in ['node1', 'protein1']), df.columns[0])
    col2 = next((c for c in df.columns if c in ['node2', 'protein2']), df.columns[1])

    all_nodes = pd.concat([df[col1], df[col2]]).unique()
    node_mapping = {name: i for i, name in enumerate(all_nodes)}
    idx_to_name = {i: name for i, name in enumerate(all_nodes)}

    src = [node_mapping[name] for name in df[col1]]
    dst = [node_mapping[name] for name in df[col2]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    num_nodes = len(all_nodes)

    # Critical: One-Hot features to give GNN identity info
    x = torch.eye(num_nodes)

    return Data(x=x, edge_index=edge_index), num_nodes, idx_to_name


# ==========================================
# 3. Helper: Metrics Calculation
# ==========================================

def get_metrics_for_scores(scores_df, test_mask_flat):
    """
    Calculates Threshold and Top-K metrics given a DataFrame of predictions.
    scores_df must have columns: ['score', 'is_true']
    """
    # 1. Threshold Analysis
    threshold_results = []
    # Dynamic thresholds based on score distribution
    max_score = scores_df['score'].max()
    thresholds = [0.5, 0.7, 0.9, 0.95]
    if max_score > 1.0:  # For CN which outputs integers
        thresholds = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35]

    for th in thresholds:
        subset = scores_df[scores_df['score'] >= th]
        if len(subset) == 0:
            continue
        tp = subset['is_true'].sum()
        precision = tp / len(subset)
        threshold_results.append((th, precision, len(subset)))

    # 2. Top-K Analysis
    top_k_results = []
    for k in [10, 20, 50, 100]:
        top_subset = scores_df.head(k)
        hits = top_subset['is_true'].sum()
        precision = hits / k
        top_k_results.append((k, precision, hits))

    return threshold_results, top_k_results


# ==========================================
# 4. Main Comparison Logic
# ==========================================

def run_detailed_comparison(file_path):
    # Load and Split
    data, num_nodes, idx_to_name = load_data(file_path)
    transform = T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True, split_labels=True,
                                  add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device)

    # -------------------------------------------------
    # A. Train GNN (VGAE)
    # -------------------------------------------------
    print("\nTraining GNN (VGAE)...")
    model = VGAE(VariationalGCNEncoder(num_nodes, 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index) + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

    # -------------------------------------------------
    # B. Generate Scores for ALL pairs (GNN)
    # -------------------------------------------------
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        # Calculate prob for every pair: sigmoid(Z * Z^t)
        gnn_adj = torch.sigmoid(torch.matmul(z, z.t()))

    # -------------------------------------------------
    # C. Generate Scores for ALL pairs (Common Neighbors)
    # -------------------------------------------------
    print("Calculating Common Neighbors...")
    # 1. Convert edge_index to Dense Adjacency Matrix
    adj_matrix = to_dense_adj(train_data.edge_index, max_num_nodes=num_nodes)[0]
    # 2. Matrix Multiplication (A^2 gives number of common neighbors)
    cn_adj = torch.matmul(adj_matrix, adj_matrix)

    # -------------------------------------------------
    # D. Prepare Evaluation Data
    # -------------------------------------------------
    # Create masks
    train_mask = adj_matrix.bool()

    # Test mask (Ground Truth)
    test_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool).to(device)
    test_pos = test_data.pos_edge_label_index.to(device)
    test_mask[test_pos[0], test_pos[1]] = True

    # Gather predictions for valid pairs (Upper triangle, not in train)
    gnn_preds = []
    cn_preds = []

    # Iterate upper triangle
    rows, cols = torch.triu_indices(num_nodes, num_nodes, offset=1)

    for r, c in zip(rows, cols):
        if train_mask[r, c]: continue  # Skip training edges

        is_true = test_mask[r, c].item()

        # GNN Score
        gnn_preds.append({
            'score': gnn_adj[r, c].item(),
            'is_true': is_true
        })

        # CN Score (Normalize? No, raw count is fine for ranking)
        cn_preds.append({
            'score': cn_adj[r, c].item(),
            'is_true': is_true
        })

    df_gnn = pd.DataFrame(gnn_preds).sort_values('score', ascending=False)
    df_cn = pd.DataFrame(cn_preds).sort_values('score', ascending=False)

    # -------------------------------------------------
    # E. Comparison Output
    # -------------------------------------------------
    gnn_th, gnn_k = get_metrics_for_scores(df_gnn, None)
    cn_th, cn_k = get_metrics_for_scores(df_cn, None)

    print("\n" + "=" * 60)
    print(f"{'HEAD-TO-HEAD RESULTS':^60}")
    print("=" * 60)

    print("\n🏆 Metric 1: Top-K Precision Comparison")
    print(f"{'K':<5} | {'GNN Precision':<15} | {'CN Precision':<15} | {'Diff'}")
    print("-" * 50)

    for (k, p_gnn, _), (_, p_cn, _) in zip(gnn_k, cn_k):
        diff = p_gnn - p_cn
        marker = "🟢 GNN Win" if diff > 0 else "⚪ Tie/Loss"
        print(f"{k:<5} | {p_gnn:.4f}          | {p_cn:.4f}          | {marker}")

    print("\n📊 Metric 2: Confidence Analysis")
    print("(Note: CN uses raw counts 1,2,3... GNN uses probability 0-1)")

    print("\n--- GNN Performance (Thresholds) ---")
    print(f"{'Thresh':<8} | {'Precision':<10} | {'Count'}")
    for th, p, c in gnn_th:
        print(f"{th:<8} | {p:.4f}     | {c}")

    print("\n--- Common Neighbor Performance (Counts) ---")
    print(f"{'Min Neighbors':<15} | {'Precision':<10} | {'Count'}")
    for th, p, c in cn_th:
        print(f"{th:<15} | {p:.4f}     | {c}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("If GNN Precision@K is close to or better than CN, the Deep Learning model")
    print("is successfully learning the topology. Look for 'False Positives' in GNN")
    print("that are NOT in CN - those are the non-trivial biological discoveries!")


if __name__ == "__main__":
    FILE_NAME = 'string_interaction_physical.tsv'
    try:
        run_detailed_comparison(FILE_NAME)
    except Exception as e:
        print(f"Error: {e}")