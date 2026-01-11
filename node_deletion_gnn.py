import torch
import pandas as pd
import numpy as np
import os
import random
from torch_geometric.nn import VGAE, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# ==========================================
# 1. Model Definition
# ==========================================
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Note: GCN relies on neighbors. New proteins have no neighbors!
        # The model will learn to rely on the Self Loop (its own features)
        # in the first layer for these isolated nodes.
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# ==========================================
# 2. Node Level Split (Inductive Split)
# ==========================================
def node_level_split(data, val_ratio=0.2, seed=42):
    """
    Splits the graph by REMOVING nodes entirely to simulate 'Cold Start'.
    
    1. Selects 'val_ratio' of nodes to be Test Nodes (Hidden).
    2. Removes these nodes and ALL their edges from the Training Set.
    3. The Training Set preserves the structure of the remaining graph.
    
    Returns:
        train_edge_index: Edges between training nodes (Structure preserved)
        test_node_indices: Indices of the removed nodes
        test_edge_index: The hidden edges connecting Test Nodes to Training Nodes
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_nodes = data.num_nodes
    all_node_indices = list(range(num_nodes))
    random.shuffle(all_node_indices)
    
    # 1. Select Nodes to Hide (Test Nodes)
    num_test = int(num_nodes * val_ratio)
    test_node_indices = torch.tensor(all_node_indices[:num_test], dtype=torch.long)
    train_node_indices = torch.tensor(all_node_indices[num_test:], dtype=torch.long)
    
    row, col = data.edge_index
    
    # 2. Identify Edges to Hide (Ground Truth for Test)
    # We want to predict edges between [Test Node] <-> [Train Node]
    # (Connecting a new protein to the existing known network)
    
    # Mask: Edges where one node is Test and the other is Train
    test_mask = (torch.isin(row, test_node_indices) & torch.isin(col, train_node_indices)) | \
                (torch.isin(row, train_node_indices) & torch.isin(col, test_node_indices))
                
    test_edge_index = data.edge_index[:, test_mask]
    
    # 3. Create Training Subgraph
    # We keep ONLY edges where BOTH nodes are in the training set.
    # This preserves the internal structure of the known network.
    
    train_mask = torch.isin(row, train_node_indices) & torch.isin(col, train_node_indices)
    train_edge_index = data.edge_index[:, train_mask]
    
    return train_edge_index, test_node_indices, test_edge_index

# ==========================================
# 3. Data Loading
# ==========================================
def load_data(file_path):
    print(f"--- Loading data from {file_path} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
        
    df = pd.read_csv(file_path, sep='\t')
    # Clean column names (remove # and spaces)
    df.rename(columns={col: col.lstrip('#').strip() for col in df.columns}, inplace=True)
    
    col1 = next((c for c in df.columns if c in ['node1', 'protein1', 'item_id_a']), df.columns[0])
    col2 = next((c for c in df.columns if c in ['node2', 'protein2', 'item_id_b']), df.columns[1])
    
    all_nodes = pd.concat([df[col1], df[col2]]).unique()
    node_mapping = {name: i for i, name in enumerate(all_nodes)}
    idx_to_name = {i: name for i, name in enumerate(all_nodes)}
    
    src = [node_mapping[name] for name in df[col1]]
    dst = [node_mapping[name] for name in df[col2]]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)
    
    num_nodes = len(all_nodes)
    
    # --- FEATURES ---
    # Currently using One-Hot Encoding (IDs).
    # TODO: For real Inductive Learning, replace this with actual biological features 
    # (e.g., Sequence Embeddings) so the model can generalize to new nodes.
    x = torch.eye(num_nodes)
    
    return Data(x=x, edge_index=edge_index), num_nodes, idx_to_name

# ==========================================
# 4. Main Analysis Logic
# ==========================================
def run_inductive_analysis(file_path):
    data, num_nodes, idx_to_name = load_data(file_path)
    
    print("\n--- Performing Inductive Split (Deleting Nodes) ---")
    # Hide 20% of the proteins
    train_edge_index, test_node_indices, hidden_edges = node_level_split(data, val_ratio=0.2)
    
    print(f"Total Nodes: {num_nodes}")
    print(f"Training Edges: {train_edge_index.shape[1]}")
    print(f"Hidden (Test) Nodes: {len(test_node_indices)}")
    print(f"Hidden Edges to Reconstruct: {hidden_edges.shape[1] // 2}") 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create training data object (Partial Graph)
    # The model sees all nodes (features) but only training edges.
    train_data = Data(x=data.x, edge_index=train_edge_index).to(device)
    
    # -------------------------------------------------
    # A. Training (On the partial graph)
    # -------------------------------------------------
    print("\nTraining GNN on partial graph...")
    # Increase hidden dimension to 32 to capture more complex patterns
    model = VGAE(VariationalGCNEncoder(num_nodes, 32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        # The model encodes using ONLY the visible training edges
        z = model.encode(train_data.x, train_data.edge_index)
        
        # We calculate loss based on reconstructing the training edges
        loss = model.recon_loss(z, train_data.edge_index) + (1/num_nodes)*model.kl_loss()
        loss.backward()
        optimizer.step()
        
    # -------------------------------------------------
    # B. The Test: Reconstructing Deleted Proteins
    # -------------------------------------------------
    print("\n🧪 The 'Cold Start' Test: Reconstructing Deleted Proteins...")
    model.eval()
    
    with torch.no_grad():
        # Get embeddings for all nodes based on the partial graph
        z = model.encode(train_data.x, train_data.edge_index)
        
        # Calculate full probability matrix (All vs All)
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))
        
    # Evaluate performance on the Hidden Nodes
    print(f"\nEvaluating {len(test_node_indices)} hidden proteins...")
    
    # Create set for fast lookup of true hidden edges
    hidden_edges_set = set()
    rows, cols = hidden_edges.cpu().numpy()
    for u, v in zip(rows, cols):
        hidden_edges_set.add(tuple(sorted((u, v))))

    print(f"\n{'Protein Name':<15} | {'True Edges':<10} | {'Top-10 Predicted Partners (V=Correct)'}")
    print("-" * 80)
    
    # Evaluate a sample of the test nodes (e.g., first 10)
    for test_node_idx in test_node_indices[:10]:
        test_node_idx = test_node_idx.item()
        
        # Get scores for this test node against all other nodes
        scores = adj_pred[test_node_idx]
        
        # We only care about connections to the Training Set (Existing Network)
        # So we mask out connections to other Test nodes or itself
        scores[test_node_indices] = -1 
        
        # Get Top 10 Predictions
        top_k_vals, top_k_indices = torch.topk(scores, 10)
        
        # Check correctness
        found_count = 0
        predicted_partners = []
        
        for partner_idx in top_k_indices:
            partner_idx = partner_idx.item()
            pair = tuple(sorted((test_node_idx, partner_idx)))
            
            is_real = pair in hidden_edges_set
            if is_real:
                found_count += 1
                
            predicted_partners.append(f"{idx_to_name[partner_idx]}({'V' if is_real else 'X'})")
            
        # Count total true edges this node had (to calculate recall manually if needed)
        # (Approximation based on hidden_edges subset)
        
        print(f"{idx_to_name[test_node_idx]:<15} | Found {found_count}    | {', '.join(predicted_partners[:3])}...")

    print("\n" + "="*60)
    print("SUMMARY OF INDUCTIVE LEARNING")
    print("This simulates discovering a NEW protein based solely on its properties.")
    print("If you see 'V' marks above, the model successfully reconstructed the edges.")
    print("NOTE: With One-Hot encoding, performance might be random for strictly new nodes.")
    print("      To improve, replace 'x' with biological sequence embeddings.")

if __name__ == "__main__":
    FILE_NAME = 'string_interaction_physical.tsv'
    try:
        run_inductive_analysis(FILE_NAME)
    except Exception as e:
        print(f"Error: {e}")