import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import numpy as np
from itertools import combinations

from extract_proteins_representations import load_embeddings
import utils


class PPIClassifier(pl.LightningModule):
    def __init__(self, embedding_dim, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Added for better regularization
            nn.Linear(128, 1)  # No Sigmoid here!
        )
        
        # Use WithLogits for numerical stability
        self.criterion = nn.BCEWithLogitsLoss() 
        self.train_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        
    def forward(self, x):
        return self.layers(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        
        # Apply sigmoid only for metric calculation
        preds = torch.sigmoid(logits)
        self.train_acc(preds, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.float())
        
        preds = torch.sigmoid(logits)
        self.val_f1(preds, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Added weight_decay to handle high-dimensional ESM-2 input
        return optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)


def prepare_tensor_data(data_list, embeddings):
    """
    Convert list of (u, v, label) to Tensor dataset.
    """
    X = []
    y = []
    
    missing_count = 0
    for u, v, label in data_list:
        if u in embeddings and v in embeddings:
            emb_u = embeddings[u]
            emb_v = embeddings[v]
            pair_emb = torch.cat([emb_u, emb_v])
            X.append(pair_emb)
            y.append(label)
        else:
            missing_count += 1
            
    if missing_count > 0:
        print(f"Warning: {missing_count} pairs skipped due to missing embeddings.")
        
    if len(X) == 0:
        # Debugging info
        print("Error: No valid pairs found. checking mismatch...")
        if len(data_list) > 0:
            print(f"Sample data pair: {data_list[0]}")
            u, v, _ = data_list[0]
            print(f"Node '{u}' in embeddings? {u in embeddings}")
            print(f"Node '{v}' in embeddings? {v in embeddings}")
            if len(embeddings) > 0:
                print(f"Sample embedding key: {list(embeddings.keys())[0]}")
        raise ValueError("X list is empty. Cannot stack tensors.")
            
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def main():
    # Load embeddings
    embeddings = load_embeddings("protein_embeddings.pkl")
    print(f"Loaded {len(embeddings)} protein embeddings.")
    
    # Convert dict values to tensors if they aren't already
    for protein_id in embeddings:
        if not isinstance(embeddings[protein_id], torch.Tensor):
            embeddings[protein_id] = torch.tensor(embeddings[protein_id], dtype=torch.float32)
    
    embedding_dim = embeddings[list(embeddings.keys())[0]].shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    
    # Load and split graph
    G = utils.load_graph()
    
    # Ensure Graph uses String IDs to match Embeddings (Fix for ID mismatch)
    df = utils.pd.read_csv("string_interactions_short.tsv", sep='\t')
    df.columns = df.columns.str.lstrip('#')
    if 'node1_string_id' in df.columns:
        id_map = utils.pd.Series(df['node1_string_id'].values, index=df['node1']).to_dict()
        id_map.update(utils.pd.Series(df['node2_string_id'].values, index=df['node2']).to_dict())
        G = utils.nx.relabel_nodes(G, id_map)
        print("Relabeled graph nodes to String IDs.")
    
    # Use the new split function with balanced negatives (Vertex Split)
    train_data, val_data, test_data = utils.split_data_semi_inductive(
        G, test_ratio=0.2, val_ratio=0.1, rnd_seed=42
    )

    # Convert to Tensors
    print("Preparing Tensor Datasets...")
    X_train, y_train = prepare_tensor_data(train_data, embeddings)

    # EXP
    # Randomize labels for sanity check
    # print("Randomizing training labels...")
    # y_train = y_train[torch.randperm(len(y_train))]
    # EXP

    X_val, y_val = prepare_tensor_data(val_data, embeddings)
    X_test, y_test = prepare_tensor_data(test_data, embeddings)

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)
    
    # Model
    model = PPIClassifier(embedding_dim=embedding_dim, lr=1e-4)
    
    # Trainer
    trainer = pl.Trainer(max_epochs=15, enable_progress_bar=True)
    trainer.fit(model, train_loader, val_loader)
    
    # --- Evaluation ---
    print("\n--- Test Set Evaluation ---")
    model.eval()
    
    logits_all = []
    y_true_all = []
    
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            logits_all.extend(logits.tolist())
            y_true_all.extend(y.tolist())
            
    y_probs = torch.sigmoid(torch.tensor(logits_all)).numpy()
    y_true = np.array(y_true_all)
    y_pred_binary = (y_probs > 0.5).astype(int)
    
    # Metrics
    tp = np.sum((y_pred_binary == 1) & (y_true == 1))
    fp = np.sum((y_pred_binary == 1) & (y_true == 0))
    tn = np.sum((y_pred_binary == 0) & (y_true == 0))
    fn = np.sum((y_pred_binary == 0) & (y_true == 1))
    
    acc, prec, rec, f1 = utils.calculate_metrics(tp, fp, tn, fn)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot ROC using utils (requires restructuring data)
    # We construct the dictionaries expected by utils.plot_roc_curve
    predictions_dict = {}
    test_edges_mock = []       # Treated as "positives" in utils.plot_roc_curve
    non_train_edges_mock = []  # Universe of evaluation edges
    
    for i, (u, v, label) in enumerate(test_data):
        prob = y_probs[i]
        edge = tuple(sorted((u, v)))
        predictions_dict[edge] = prob
        non_train_edges_mock.append(edge)
        
        if label == 1:
            test_edges_mock.append((u, v, {}))
            
    print("\nPlotting ROC curve...")
    auc_score = utils.plot_roc_curve(
        predictions_dict, 
        test_edges_mock, 
        non_train_edges_mock, 
        model_name="PPIClassifier_Seq"
    )
    if auc_score:
        print(f"AUC: {auc_score:.4f}")


if __name__ == '__main__':
    main()