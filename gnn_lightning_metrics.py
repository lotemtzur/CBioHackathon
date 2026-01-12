# ppi_vgae_coldstart_metrics.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import VGAE, GCNConv
from torch.utils.data import DataLoader, TensorDataset

# Assuming these exist in your environment
from extract_proteins_representations import load_embeddings
import utils


############################################
# 0. DATA PREP & NODE SPLITTING
############################################

def load_graph_and_split_nodes(tsv_path, embeddings, test_ratio=0.2):
    """
    Loads the graph and splits NODES (not just edges) into Train and Test sets.
    This simulates a true 'Cold Start' scenario.
    """
    # 1. Load Graph
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = df.columns.str.lstrip("#")

    G_full = nx.Graph()
    for _, row in df.iterrows():
        G_full.add_edge(row["node1_string_id"], row["node2_string_id"])

    # 2. Filter nodes that have embeddings
    valid_nodes = sorted(list(set(G_full.nodes()) & set(embeddings.keys())))

    # 3. Split Nodes
    np.random.shuffle(valid_nodes)
    split_idx = int(len(valid_nodes) * (1 - test_ratio))
    train_nodes = valid_nodes[:split_idx]
    test_nodes = valid_nodes[split_idx:]

    # 4. Create Train Subgraph (for VGAE)
    G_train = G_full.subgraph(train_nodes).copy()

    # Map node IDs to 0..N for PyG
    node2idx_train = {n: i for i, n in enumerate(train_nodes)}

    edges_train = []
    for u, v in G_train.edges():
        edges_train.append((node2idx_train[u], node2idx_train[v]))
        edges_train.append((node2idx_train[v], node2idx_train[u]))  # Undirected

    edge_index_train = torch.tensor(edges_train, dtype=torch.long).t()

    # Feature matrix (Identity or Zeros for pure structure learning)
    x_train = torch.eye(len(train_nodes))  # Identity is often better for node ID learning
    # x_train = torch.zeros((len(train_nodes), 1)) # Or zeros

    data_train = Data(x=x_train, edge_index=edge_index_train)

    return data_train, train_nodes, test_nodes, G_full


############################################
# 1. MODELS
############################################

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, z_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.mu = GCNConv(hidden_dim, z_dim)
        self.logvar = GCNConv(hidden_dim, z_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        return self.mu(h, edge_index), self.logvar(h, edge_index)


class VGAE_GCN(VGAE):
    def __init__(self, in_dim, hidden_dim, z_dim):
        super().__init__(GCNEncoder(in_dim, hidden_dim, z_dim))


class ESM2ToZ(nn.Module):
    def __init__(self, esm_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.BatchNorm1d(512),  # Added Batch Norm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )

    def forward(self, x):
        return self.net(x)


############################################
# 2. TRAINING LOOPS
############################################

def train_vgae(data, epochs=200, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAE_GCN(data.num_features, 64, 32).to(device)  # Reduced dims for stability
    opt = optim.Adam(model.parameters(), lr=lr)
    data = data.to(device)

    model.train()
    for e in range(epochs):
        opt.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss = loss + (1.0 / data.num_nodes) * model.kl_loss()
        loss.backward()
        opt.step()

        if e % 50 == 0:
            print(f"[VGAE] Epoch {e:03d} | Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
    return z.detach().cpu(), model  # Return model to save if needed


def train_mlp(esm_train, z_train, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure inputs are detached
    esm_train = esm_train.detach().to(device)
    z_train = z_train.detach().to(device)

    model = ESM2ToZ(esm_train.size(1), z_train.size(1)).to(device)
    loader = DataLoader(TensorDataset(esm_train, z_train), batch_size=32, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for e in range(epochs):
        epoch_loss = 0
        for esm_b, z_b in loader:
            opt.zero_grad()
            pred = model(esm_b)
            loss = criterion(pred, z_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        if e % 20 == 0:
            print(f"[MLP] Epoch {e:03d} | MSE Loss {epoch_loss / len(loader):.4f}")

    return model.cpu()


############################################
# 3. EVALUATION
############################################

def evaluate_cold_start(mlp, z_train_structure, esm_test, test_nodes, train_nodes, G_full):
    """
    Evaluates how well the Cold Start MLP predicts edges between
    UNSEEN (Test) nodes and SEEN (Train) nodes.
    """
    print("\n--- Cold Start Evaluation ---")
    mlp.eval()

    # Metrics
    metric_acc = BinaryAccuracy()
    metric_f1 = BinaryF1Score()
    metric_prec = BinaryPrecision()
    metric_rec = BinaryRecall()

    # 1. Infer Z for test nodes using MLP (Sequence -> Structure)
    with torch.no_grad():
        z_test_predicted = mlp(esm_test)  # shape [num_test, z_dim]

    # 2. Prepare Ground Truth & Predictions
    y_true = []
    y_scores = []

    # We will test all pairs (Test_Node, Train_Node)
    # Note: For large graphs, you should sample negatives instead of doing all pairs.
    print(f"Evaluating {len(test_nodes)} test nodes against {len(train_nodes)} train nodes...")

    # Convert lists to sets for fast lookup
    train_node_set = set(train_nodes)

    for i, u in enumerate(test_nodes):
        # Get Ground Truth edges for u
        u_neighbors = set(G_full.neighbors(u))

        # Calculate scores against ALL train nodes
        # z_test_predicted[i] shape is [z_dim]
        # z_train_structure shape is [num_train, z_dim]
        # score = dot product
        scores = torch.matmul(z_train_structure, z_test_predicted[i])  # Shape [num_train]
        scores = torch.sigmoid(scores)  # Sigmoid for probability

        # Build labels
        labels = torch.zeros_like(scores)
        for j, v in enumerate(train_nodes):
            if v in u_neighbors:
                labels[j] = 1.0

        y_true.append(labels)
        y_scores.append(scores)

    # Concat all
    y_true = torch.cat(y_true)
    y_scores = torch.cat(y_scores)
    y_pred = (y_scores > 0.8).long()

    # 3. Compute Metrics
    acc = metric_acc(y_pred, y_true)
    f1 = metric_f1(y_pred, y_true)
    prec = metric_prec(y_pred, y_true)
    rec = metric_rec(y_pred, y_true)

    # Sklearn for AUC (Torchmetrics AUC can be finicky with flat tensors sometimes)
    auc = roc_auc_score(y_true.numpy(), y_scores.numpy())
    ap = average_precision_score(y_true.numpy(), y_scores.numpy())

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Avg Prec:  {ap:.4f}")


############################################
# 4. MAIN
############################################

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    embeddings = load_embeddings("protein_embeddings.pkl")

    # 1. LOAD & SPLIT NODES
    # This ensures we have a true 'Test Set' of proteins the VGAE never sees
    data_train, train_nodes, test_nodes, G_full = load_graph_and_split_nodes(
        "string_interactions_short.tsv", embeddings, test_ratio=0.2
    )

    print(f"Train Nodes: {len(train_nodes)} | Test Nodes: {len(test_nodes)}")

    # Prepare ESM matrices
    # IMPORTANT: .detach() to prevent graph re-traversal errors
    esm_train = torch.stack([embeddings[n] for n in train_nodes]).detach()
    esm_test = torch.stack([embeddings[n] for n in test_nodes]).detach()

    # 2. TRAIN VGAE (Structure Learning on Train Nodes)
    print("\nTraining VGAE (Structure Learning)...")
    z_train_structure, vgae_model = train_vgae(data_train)

    # 3. TRAIN MLP (Mapping ESM -> Z on Train Nodes)
    print("\nTraining MLP (Sequence -> Structure Mapping)...")
    mlp_model = train_mlp(esm_train, z_train_structure)

    # 4. EVALUATE (Cold Start on Test Nodes)
    evaluate_cold_start(mlp_model, z_train_structure, esm_test, test_nodes, train_nodes, G_full)

    # Save
    torch.save(mlp_model.state_dict(), "coldstart_mlp.pt")
    print("\nModel saved.")


if __name__ == "__main__":
    main()