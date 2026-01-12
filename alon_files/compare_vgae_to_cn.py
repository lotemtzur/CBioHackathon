import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.metrics import average_precision_score, precision_recall_curve

# PyTorch Geometric
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, to_networkx


# ==========================================
# 1. VGAE Encoder
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
# 2. Common Neighbors Baseline
# ==========================================
class CommonNeighborsPredictor:
    def __init__(self, edge_index):
        self.adj = defaultdict(set)
        edges = edge_index.cpu().numpy()
        for i in range(edges.shape[1]):
            u, v = edges[0, i], edges[1, i]
            self.adj[u].add(v)
            self.adj[v].add(u)

    def predict_score(self, src, dst):
        scores = []
        for u, v in zip(src, dst):
            u, v = u.item(), v.item()
            scores.append(len(self.adj[u].intersection(self.adj[v])))
        return torch.tensor(scores, dtype=torch.float)


# ==========================================
# 3. Precision@K
# ==========================================
def precision_at_k(scores, labels, k):
    idx = np.argsort(scores)[::-1][:k]
    return labels[idx].mean()


# ==========================================
# 4. Load Data
# ==========================================
def load_data(file_path):
    print(f"--- Loading {file_path} ---")
    df = pd.read_csv(file_path, sep="\t")
    df.rename(columns={c: c.lstrip("#") for c in df.columns}, inplace=True)

    col1 = df.columns[0]
    col2 = df.columns[1]

    nodes = pd.concat([df[col1], df[col2]]).unique()
    node_map = {n: i for i, n in enumerate(nodes)}

    src = [node_map[n] for n in df[col1]]
    dst = [node_map[n] for n in df[col2]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    x = torch.nn.init.xavier_uniform_(torch.empty(len(nodes), 16))

    return Data(x=x, edge_index=edge_index), len(nodes)


# ==========================================
# 5. Main Experiment (PR–AUC only)
# ==========================================
def run_experiment(file_path):
    data, num_nodes = load_data(file_path)

    # Graph clustering (sanity check)
    import networkx as nx
    G = to_networkx(data)
    print(f"Clustering coefficient: {nx.average_clustering(G):.4f}")

    # Unified split
    transform = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.15,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False
    )
    train_data, _, test_data = transform(data)

    print(f"Train edges: {train_data.edge_index.size(1)}")
    print(f"Test positives: {test_data.pos_edge_label_index.size(1)}")

    # ==============================
    # Common Neighbors (PR–AUC)
    # ==============================
    print("\n1️⃣ Common Neighbors")

    cn_model = CommonNeighborsPredictor(train_data.edge_index)

    pos = test_data.pos_edge_label_index
    neg = test_data.neg_edge_label_index

    cn_pos_scores = cn_model.predict_score(pos[0], pos[1])
    cn_neg_scores = cn_model.predict_score(neg[0], neg[1])

    cn_scores = torch.cat([cn_pos_scores, cn_neg_scores]).numpy()
    cn_labels = np.concatenate([
        np.ones(len(cn_pos_scores)),
        np.zeros(len(cn_neg_scores))
    ])

    cn_ap = average_precision_score(cn_labels, cn_scores)
    cn_prec, cn_rec, _ = precision_recall_curve(cn_labels, cn_scores)

    print(f"PR-AUC (AP): {cn_ap:.4f}")
    for k in [10, 50, 100, 500]:
        print(f"Precision@{k}: {precision_at_k(cn_scores, cn_labels, k):.3f}")

    # ==============================
    # VGAE (PR–AUC)
    # ==============================
    print("\n2️⃣ VGAE")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAE(VariationalGCNEncoder(16, 16)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_data = train_data.to(device)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index) \
               + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        pos_scores = model.decoder(z, pos.to(device)).cpu()
        neg_scores = model.decoder(z, neg.to(device)).cpu()

    vgae_scores = torch.cat([pos_scores, neg_scores]).numpy()
    vgae_labels = np.concatenate([
        np.ones(len(pos_scores)),
        np.zeros(len(neg_scores))
    ])

    vgae_ap = average_precision_score(vgae_labels, vgae_scores)
    vgae_prec, vgae_rec, _ = precision_recall_curve(vgae_labels, vgae_scores)

    print(f"PR-AUC (AP): {vgae_ap:.4f}")
    for k in [10, 50, 100, 500]:
        print(f"Precision@{k}: {precision_at_k(vgae_scores, vgae_labels, k):.3f}")

    # ==============================
    # Precision–Recall Plot
    # ==============================
    plt.figure(figsize=(7, 5))
    plt.plot(cn_rec, cn_prec, label=f"CN (AP={cn_ap:.3f})")
    plt.plot(vgae_rec, vgae_prec, label=f"VGAE (AP={vgae_ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Link Prediction)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    FILE_NAME = "string_interactions_short.tsv"
    run_experiment(FILE_NAME)
