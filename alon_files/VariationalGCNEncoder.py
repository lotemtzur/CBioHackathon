import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from torch_geometric.nn import VGAE, GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# ==========================================
# 1. הגדרת המודל (Encoder: GCN)
# ==========================================

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # שכבת GCN ראשונה - משתפת מידע בין שכנים
        self.conv1 = GCNConv(in_channels, 2 * out_channels)

        # שכבות GCN שניות - לומדות את הממוצע והשונות להסתברות
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # שלב 1: אקטיבציה ראשונית
        x = self.conv1(x, edge_index).relu()
        # שלב 2: החזרת פרמטרים להתפלגות (Latent Space)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# ==========================================
# 2. פונקציה לטעינת הדאטה מהקובץ שלך
# ==========================================

def load_string_data(file_path):
    print(f"--- Loading data from: {file_path} ---")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Error: The file '{file_path}' was not found in the directory.")

    # טעינת הקובץ (הנחה: הפרדה בטאבים)
    df = pd.read_csv(file_path, sep='\t')

    # --- תיקון שמות עמודות ---
    # הסרת רווחים וסימן '#' אם קיים (נפוץ בקבצי STRING כמו #node1)
    df.rename(columns={col: col.lstrip('#').strip() for col in df.columns}, inplace=True)

    print(f"✅ Columns found: {list(df.columns)}")

    # זיהוי אוטומטי של עמודות המקור והיעד
    # בודקים וריאציות נפוצות: node1/node2, protein1/protein2
    potential_source_cols = ['node1', 'protein1', 'item_id_a']
    potential_target_cols = ['node2', 'protein2', 'item_id_b']

    col1 = next((c for c in df.columns if c in potential_source_cols), df.columns[0])
    col2 = next((c for c in df.columns if c in potential_target_cols), df.columns[1])

    print(f"✅ Using columns: Source='{col1}', Target='{col2}'")

    # יצירת מיפוי (Mapping) משמות חלבונים למספרים (Indices)
    all_nodes = pd.concat([df[col1], df[col2]]).unique()
    node_mapping = {name: i for i, name in enumerate(all_nodes)}

    num_nodes = len(all_nodes)
    print(f"✅ Unique Proteins (Nodes): {num_nodes}")
    print(f"✅ Raw Interactions (Edges): {len(df)}")

    # המרת שמות החלבונים למספרים
    src = [node_mapping[name] for name in df[col1]]
    dst = [node_mapping[name] for name in df[col2]]

    # יצירת טנסור הקשתות (Edge Index)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    # --- יצירת Features (x) ---
    # שימוש ב-Embeddings התחלתיים ברי-למידה
    num_features = 16
    x = torch.nn.init.xavier_uniform_(torch.empty(num_nodes, num_features))

    data = Data(x=x, edge_index=edge_index)

    return data, num_nodes, num_features


# ==========================================
# 3. הכנת ה-Split והמודל
# ==========================================

FILENAME = '../string_interactions_short.tsv'

# 1. טעינת הנתונים
try:
    data, num_nodes, num_features = load_string_data(FILENAME)
except Exception as e:
    print(f"Error: {e}")
    # כדי למנוע קריסה אם הקובץ חסר, נעצור כאן בצורה מסודרת
    import sys

    sys.exit(1)

# 2. חלוקה ל-Train / Val / Test
print("--- Splitting data into Train/Val/Test ---")
transform = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    split_labels=True,
    add_negative_train_samples=False
)

train_data, val_data, test_data = transform(data)
print(f"Train edges: {train_data.edge_index.size(1)}")

# 3. אתחול המודל
out_channels = 16
encoder = VariationalGCNEncoder(num_features, out_channels)
model = VGAE(encoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_data = train_data.to(device)
test_data = test_data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ==========================================
# 4. לולאת האימון (Training Loop)
# ==========================================

def train():
    model.train()
    optimizer.zero_grad()

    z = model.encode(train_data.x, train_data.edge_index)

    loss = model.recon_loss(z, train_data.pos_edge_label_index) + \
           (1 / num_nodes) * model.kl_loss()

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_model(data_set):  # שם חדש לפונקציה כדי למנוע התנגשות עם Pytest
    model.eval()
    z = model.encode(train_data.x, train_data.edge_index)
    return model.test(z, data_set.pos_edge_label_index, data_set.neg_edge_label_index)


print("\n--- Starting Training ---")
for epoch in range(1, 201):
    loss = train()

    if epoch % 20 == 0:
        # שימוש בפונקציה החדשה evaluate_model
        auc, ap = evaluate_model(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test AUC: {auc:.4f}, Test AP: {ap:.4f}')

print("\n✅ Training Complete!")
print("Final Results indicate the probability that the model correctly predicts missing links.")