import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.nn import VGAE, GCNConv
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


# ==========================================
# 1. הגדרת המודל (אותו מודל VGAE)
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
# 2. פונקציית טעינת דאטה (עם One-Hot)
# ==========================================
def load_data(file_path):
    print(f"--- Loading data from {file_path} ---")
    df = pd.read_csv(file_path, sep='\t')
    df.rename(columns={col: col.lstrip('#').strip() for col in df.columns}, inplace=True)

    col1 = next((c for c in df.columns if c in ['node1', 'protein1']), df.columns[0])
    col2 = next((c for c in df.columns if c in ['node2', 'protein2']), df.columns[1])

    all_nodes = pd.concat([df[col1], df[col2]]).unique()
    node_mapping = {name: i for i, name in enumerate(all_nodes)}

    # שמירת המיפוי כדי שנוכל להדפיס שמות חלבונים בסוף
    idx_to_prot = {i: name for name, i in node_mapping.items()}

    src = [node_mapping[name] for name in df[col1]]
    dst = [node_mapping[name] for name in df[col2]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = to_undirected(edge_index)

    num_nodes = len(all_nodes)
    x = torch.eye(num_nodes)  # One-Hot Encoding

    return Data(x=x, edge_index=edge_index), num_nodes, idx_to_prot


# ==========================================
# 3. ניתוח מתקדם (החלק החדש)
# ==========================================
def detailed_analysis(model, data, train_data, test_data, idx_to_prot, device):
    model.eval()
    with torch.no_grad():
        # 1. יצירת ה-Embeddings הסופיים (Z)
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))

        # 2. חישוב מטריצת הסתברויות מלאה (All Pairs)
        # מכפילים את Z בעצמו ומפעילים Sigmoid כדי לקבל הסתברות בין 0 ל-1 לכל זוג
        prob_adj = torch.sigmoid(torch.matmul(z, z.t()))

        # המרה ל-CPU ו-Numpy לעיבוד נוח
    prob_adj = prob_adj.cpu().numpy()

    # 3. יצירת מסכות (Masks)
    # אנחנו צריכים לדעת איזה זוגות היו ב-Train (להתעלם), איזה ב-Test (האמת), ואיזה בכלל לא קיימים
    num_nodes = data.num_nodes

    # מסכת Train (מה שכבר ראינו)
    train_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    train_edges = train_data.edge_index.cpu()
    train_mask[train_edges[0], train_edges[1]] = True

    # מסכת Test (האמת שאנחנו מחפשים - Ground Truth)
    test_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    test_pos_edges = test_data.pos_edge_label_index.cpu()
    test_mask[test_pos_edges[0], test_pos_edges[1]] = True

    # איסוף כל התחזיות הרלוונטיות (שאינן Train ואינן האלכסון עצמו)
    predictions = []

    print("\n--- Generating Predictions for all pairs ---")
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # רק משולש עליון (בלי כפילויות i-j, j-i)
            if train_mask[i, j]:
                continue  # מדלגים על מה שכבר למדנו באימון

            score = prob_adj[i, j]
            is_true_link = test_mask[i, j].item()

            predictions.append({
                'node1': idx_to_prot[i],
                'node2': idx_to_prot[j],
                'score': score,
                'is_true': is_true_link
            })

    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values(by='score', ascending=False)

    # ---------------------------------------------------------
    # ניתוח 1: Precision & Recall לפי ספים (Thresholds)
    # ---------------------------------------------------------
    print("\n📊 Metric 1: Threshold Analysis (Iterating over edges)")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Predicted edges':<15}")
    print("-" * 65)

    for threshold in [0.5, 0.7, 0.8, 0.9, 0.95]:
        subset = df_pred[df_pred['score'] >= threshold]

        tp = subset['is_true'].sum()  # כמה צדקנו
        fp = len(subset) - tp  # כמה טעינו
        fn = test_mask.sum().item() / 2 - tp  # כמה פספסנו (חלקי 2 כי המסכה סימטרית)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{threshold:<10} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}     | {len(subset):<15}")

    # ---------------------------------------------------------
    # ניתוח 2: Top-K Analysis (הכי חשוב לביולוגים!)
    # ---------------------------------------------------------
    print("\n🏆 Metric 2: Top-K Precision (What allows us to give a list to biologists)")
    print(f"{'K':<5} | {'Precision@K':<15} | {'True Links Found'}")
    print("-" * 45)

    for k in [10, 20, 50, 100, 200]:
        top_k = df_pred.head(k)
        hits = top_k['is_true'].sum()
        prec_k = hits / k
        print(f"{k:<5} | {prec_k:.4f}          | {hits}")

    # ---------------------------------------------------------
    # הצגת הדובדבן שבקצפת: המועמדים החדשים
    # ---------------------------------------------------------
    print("\n🔍 Top 5 False Positives (Potential NEW Discoveries!)")
    print("אלו קשרים שהמודל בטוח בהם מאוד (High Score), אבל לא היו ב-Test Set.")
    print("בהאקתון - אלו האינטראקציות שנלך לחפש ב-PubMed!")

    # לוקחים את אלו שהם False (is_true == False) אבל עם הציון הכי גבוה
    new_discoveries = df_pred[df_pred['is_true'] == False].head(5)
    print(new_discoveries[['node1', 'node2', 'score']])


# ==========================================
# 4. הרצה ראשית
# ==========================================
if __name__ == "__main__":
    FILE_NAME = 'string_interactions_short.tsv'

    # 1. טעינה ואימון מהיר
    data, num_nodes, idx_to_prot = load_data(FILE_NAME)
    transform = T.RandomLinkSplit(num_val=0.05, num_test=0.15, is_undirected=True, split_labels=True,
                                  add_negative_train_samples=False)
    train_data, val_data, test_data = transform(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(VariationalGCNEncoder(num_nodes, 32)).to(device)  # שימוש ב-32 ממדים
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_data = train_data.to(device)

    print("Training model...")
    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index) + (1 / num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()

    # 2. הרצת הניתוח המפורט
    detailed_analysis(model, data, train_data, test_data, idx_to_prot, device)