import numpy as np
import networkx as nx
from itertools import combinations
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
        
class RandomWalkBaseline:
    
    def __init__(self, G, alpha=0.15):
        self.G = G
        self.alpha = alpha
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.R = self._compute_rwr_matrix()
        
    def _compute_rwr_matrix(self):
        adj_matrix = nx.to_numpy_array(self.G, nodelist=self.nodes)
        n = len(self.nodes)
        
        # Normalize adjacency matrix to get transition probabilities
        row_sums = adj_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        W = adj_matrix / row_sums[:, np.newaxis]
        
        # Solve steady-state equation: R = alpha * (I - (1 - alpha) * W^T)^-1
        I = np.eye(n)
        A_mat = I - (1 - self.alpha) * W.T
        R = self.alpha * np.linalg.inv(A_mat)
        
        return R

    def predict(self, threshold=None):
        """Predict edges based on Random Walk with Restart scores."""
        predictions = {}
        for u, v in combinations(self.nodes, 2):
            if not self.G.has_edge(u, v):
                idx_u = self.node_to_idx[u]
                idx_v = self.node_to_idx[v]
                
                # The affinity score is the average probability of reaching v from u and u from v
                score = (self.R[idx_u, idx_v] + self.R[idx_v, idx_u]) / 2
                
                if threshold is None:
                    predictions[(u, v)] = score
                else:
                    predictions[(u, v)] = score > threshold
                    
        return predictions
    
    
def tune_alpha(G_train, alphas=[0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8]):
    print("\n--- Starting Alpha Tuning ---")
    
    G_val_train, val_edges, val_non_train = utils.graph_split_rnd(
        G_train, test_ratio=0.2, rnd_seed=42
    )
    
    best_alpha = None
    best_auc = -1
    alpha_results = {}

    val_test_set = set([tuple(sorted((u, v))) for u, v, _ in val_edges])

    for a in alphas:
        model = RandomWalkBaseline(G_val_train, alpha=a)
        preds = model.predict(threshold=None)
        
        y_true = []
        y_scores = []
        
        for (u, v) in val_non_train:
            edge = tuple(sorted((u, v)))
            score = preds.get(edge, preds.get((v, u), 0))
            
            y_true.append(1 if edge in val_test_set else 0)
            y_scores.append(score)
            
        current_auc = roc_auc_score(y_true, y_scores)
        alpha_results[a] = current_auc
        print(f"Alpha: {a:.2f} | Validation AUC: {current_auc:.4f}")
        
        if current_auc > best_auc:
            best_auc = current_auc
            best_alpha = a

    print(f"Best Alpha found: {best_alpha} with AUC: {best_auc:.4f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(list(alpha_results.keys()), list(alpha_results.values()), marker='o', color='teal')
    plt.title("Alpha Optimization: Validation AUC")
    plt.xlabel("Alpha (Restart Probability)")
    plt.ylabel("AUC Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return best_alpha
    
    
def main():
    # Load graph
    G = utils.load_graph()
    G_train, test_edges, non_train_edges = utils.preserve_density_split(G, test_ratio=0.4, n_bins=10, rnd_seed=41)
    
    optimal_alpha = tune_alpha(G_train)
    
    model = RandomWalkBaseline(G_train, alpha=optimal_alpha)
    
    # Get scores (no threshold) for AUC plot
    predictions_scores = model.predict(threshold=None)
    
    # Get binary predictions for metrics calculation
    predictions_binary = model.predict(threshold=0.01)
    
    tp, fp, tn, fn = utils.compare_predictions(predictions_binary, test_edges, non_train_edges)
    acc, precision, recall, f1 = utils.calculate_metrics(tp, fp, tn, fn)
    print(f"Random Walk Baseline Performance, using alpha={model.alpha} and threshold=0.01:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Plot AUC
    print("\nPlotting ROC curve...")
    auc_score = utils.plot_roc_curve(predictions_scores, test_edges, non_train_edges, model_name="RandomWalk")
    if auc_score:
        print(f"AUC: {auc_score:.4f}")

if __name__ == '__main__':
    main()