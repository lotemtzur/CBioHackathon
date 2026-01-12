import numpy as np
import networkx as nx
from itertools import combinations
import utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import argparse


class AdamicAdarSequence:
    """
    Graph-based scoring with Sequence Features for node prediction.
    
    This algorithm predicts interactions for isolated test nodes by:
    1. Using sequence similarity to find k virtual neighbors in the training graph
    2. Computing scores through these virtual neighbors (CN or Adamic-Adar)
    3. Ranking potential partners by weighted scores
    """
    
    def __init__(self, G_train, embeddings, gene_to_protein_id, k=10, scoring_method='adamic_adar'):
        """
        Args:
            G_train: NetworkX graph with training edges only
            embeddings: dict mapping protein_id -> embedding vector
            gene_to_protein_id: dict mapping gene_name -> protein_id
            k: number of virtual neighbors to use for each test node
            scoring_method: 'adamic_adar' or 'common_neighbors'
        """
        self.G_train = G_train
        self.embeddings = embeddings
        self.gene_to_protein_id = gene_to_protein_id
        self.k = k
        self.scoring_method = scoring_method
        
        # Pre-compute degrees for Adamic-Adar weights
        self.degrees = dict(G_train.degree())
        
        # Cache of training node embeddings for efficiency
        self.train_nodes = list(G_train.nodes())
        self.train_embeddings = {}
        for node in self.train_nodes:
            protein_id = gene_to_protein_id.get(node)
            if protein_id and protein_id in embeddings:
                self.train_embeddings[node] = embeddings[protein_id]
        
        print(f"Initialized {scoring_method.upper()} with {len(self.train_embeddings)} training nodes with embeddings")
    
    def find_virtual_neighbors(self, test_node):
        """
        Find k most similar training nodes by sequence similarity.
        
        Args:
            test_node: gene name of test node
            
        Returns:
            list of (train_node, similarity_score) tuples
        """
        test_protein_id = self.gene_to_protein_id.get(test_node)
        if not test_protein_id or test_protein_id not in self.embeddings:
            return []
        
        test_embedding = self.embeddings[test_protein_id]
        
        # Compute similarities to all training nodes
        similarities = []
        for train_node, train_embedding in self.train_embeddings.items():
            sim = utils.cosine_similarity(test_embedding, train_embedding)
            similarities.append((train_node, sim))
        
        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.k]
    
    def compute_common_neighbors_weight(self, node_v, node_partner):
        """
        Compute Common Neighbors weight between two nodes in training graph.
        CN weight = edge_weight if edge exists, else 0
        
        Args:
            node_v: virtual neighbor node
            node_partner: potential partner node
            
        Returns:
            Common Neighbors weight (float)
        """
        if not self.G_train.has_edge(node_v, node_partner):
            return 0.0
        
        # Return the edge weight (or 1.0 if no weight)
        edge_data = self.G_train.get_edge_data(node_v, node_partner)
        return edge_data.get('weight', 1.0) if edge_data else 1.0
    
    def compute_adamic_adar_weight(self, node_v, node_partner):
        """
        Compute Adamic-Adar weight between two nodes in training graph.
        AA weight = 1 / log(degree(v)) if edge exists, else 0
        
        Args:
            node_v: virtual neighbor node
            node_partner: potential partner node
            
        Returns:
            Adamic-Adar weight (float)
        """
        if not self.G_train.has_edge(node_v, node_partner):
            return 0.0
        
        degree_v = self.degrees[node_v]
        
        # Avoid log(1) = 0 and log(0) = undefined
        if degree_v <= 1:
            return 1.0  # Use 1.0 as default weight for low-degree nodes
        
        return 1.0 / np.log(degree_v)
    
    def predict_for_node(self, test_node):
        """
        Predict interaction partners for a single test node.
        
        Args:
            test_node: gene name of test node
            
        Returns:
            dict mapping partner_node -> score
        """
        # Find virtual neighbors by sequence similarity
        virtual_neighbors = self.find_virtual_neighbors(test_node)
        
        if not virtual_neighbors:
            return {}
        
        # Compute weighted Adamic-Adar scores for all potential partners
        partner_scores = defaultdict(float)
        
        for potential_partner in self.train_nodes:
            # Don't predict self-loops
            if potential_partner == test_node:
                continue
            
            score = 0.0
            for virtual_neighbor, seq_similarity in virtual_neighbors:
                # Skip if virtual neighbor is the same as potential partner
                if virtual_neighbor == potential_partner:
                    continue
                
                # Compute weight through this virtual neighbor based on scoring method
                if self.scoring_method == 'common_neighbors':
                    weight = self.compute_common_neighbors_weight(virtual_neighbor, potential_partner)
                else:  # adamic_adar
                    weight = self.compute_adamic_adar_weight(virtual_neighbor, potential_partner)
                
                # Weight by sequence similarity to virtual neighbor
                score += seq_similarity * weight
            
            if score > 0:
                partner_scores[potential_partner] = score
        
        return dict(partner_scores)
    
    def predict_for_multiple_nodes(self, test_nodes, top_k=10):
        """
        Predict interactions for multiple test nodes.
        
        Args:
            test_nodes: list of gene names (test nodes)
            top_k: number of top predictions per node
            
        Returns:
            dict mapping test_node -> list of (partner, score) tuples
        """
        predictions = {}
        
        for i, test_node in enumerate(test_nodes):
            if (i + 1) % 10 == 0:
                print(f"Processing test node {i+1}/{len(test_nodes)}...")
            
            scores = self.predict_for_node(test_node)
            
            # Sort by score and take top k
            sorted_partners = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            predictions[test_node] = sorted_partners[:top_k]
        
        return predictions


def node_level_evaluation_with_edges(model, test_nodes, test_edges, top_k=10):
    """
    Evaluate node prediction using the actual removed test edges.
    
    Args:
        model: AdamicAdarSequence model
        test_nodes: list of test node names
        test_edges: list of (u, v, data) tuples that were removed
        top_k: number of predictions per node
        
    Returns:
        precision, recall, f1_score
    """
    predictions = model.predict_for_multiple_nodes(test_nodes, top_k=top_k)
    
    # Build set of true test edges
    true_test_edges = set()
    edges_per_node = {}
    for u, v, _ in test_edges:
        edge = tuple(sorted((u, v)))
        true_test_edges.add(edge)
        # Track edges for each test node
        for node in [u, v]:
            if node in test_nodes:
                if node not in edges_per_node:
                    edges_per_node[node] = set()
                edges_per_node[node].add(edge)
    
    tp = 0  # True Positives: predicted edge that exists in test_edges
    fp = 0  # False Positives: predicted edge that doesn't exist in test_edges
    fn = len(true_test_edges)  # False Negatives: start with all test edges, subtract TPs
    
    for test_node, predicted_partners in predictions.items():
        # Check how many predictions are correct
        for partner, score in predicted_partners:
            edge = tuple(sorted((test_node, partner)))
            if edge in true_test_edges:
                tp += 1
            else:
                fp += 1
    
    fn = len(true_test_edges) - tp  # Edges we didn't predict
    
    # Calculate total possible edges to get TN
    total_test_nodes = len(test_nodes)
    total_train_nodes = len(model.train_nodes)
    total_possible_edges = total_test_nodes * total_train_nodes
    tn = total_possible_edges - tp - fp - fn  # Everything else is true negative
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1, tp, fp, tn, fn


def node_level_evaluation(model, G_full, test_nodes, top_k=10):
    """
    Evaluate node prediction using precision metrics.
    
    Args:
        model: AdamicAdarSequence model
        G_full: full graph with all edges (ground truth)
        test_nodes: list of test node names
        top_k: number of predictions per node
        
    Returns:
        precision, recall, f1_score
    """
    predictions = model.predict_for_multiple_nodes(test_nodes, top_k=top_k)
    
    total_correct = 0
    total_predictions = 0
    total_actual = 0
    
    for test_node, predicted_partners in predictions.items():
        # Ground truth: actual neighbors in full graph
        if test_node in G_full:
            actual_neighbors = set(G_full.neighbors(test_node))
            total_actual += len(actual_neighbors)
        else:
            actual_neighbors = set()
        
        # Check how many predictions are correct
        predicted_nodes = [partner for partner, score in predicted_partners]
        correct = sum(1 for p in predicted_nodes if p in actual_neighbors)
        
        total_correct += correct
        total_predictions += len(predicted_nodes)
    
    precision = total_correct / total_predictions if total_predictions > 0 else 0
    recall = total_correct / total_actual if total_actual > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def compute_auc_for_nodes(model, test_nodes, test_edges):
    """
    Compute AUC and plot ROC curve for node prediction.
    
    For each test node, we predict scores for connections to training nodes.
    We evaluate against the actual test_edges that were removed.
    
    Args:
        model: AdamicAdarSequence model
        test_nodes: list of test node names
        test_edges: list of (u, v, data) tuples that were removed
        
    Returns:
        auc_score: Area under ROC curve
    """
    from sklearn.metrics import roc_curve, auc
    
    y_true = []
    y_scores = []
    
    # Build set of true test edges (edges that were removed)
    true_test_edges = set()
    for u, v, _ in test_edges:
        true_test_edges.add(tuple(sorted((u, v))))
    
    print("\nComputing AUC across all test nodes...")
    print(f"Total test edges to predict: {len(true_test_edges)}")
    
    for i, test_node in enumerate(test_nodes):
        if (i + 1) % 10 == 0:
            print(f"  Processing node {i+1}/{len(test_nodes)} for AUC...")
        
        # Get scores for all potential partners
        scores = model.predict_for_node(test_node)
        
        if not scores:
            continue
        
        # For each TRAINING node, check if it was a removed test edge
        for partner in model.train_nodes:
            if partner == test_node:
                continue
            
            edge = tuple(sorted((test_node, partner)))
            score = scores.get(partner, 0.0)  # Default to 0 if no path through virtual neighbors
            is_test_edge = edge in true_test_edges
            
            y_true.append(1 if is_test_edge else 0)
            y_scores.append(score)
    
    if len(set(y_true)) < 2:
        print("Warning: Not enough classes for AUC calculation")
        return None
    
    print(f"Total samples: {len(y_true)}")
    print(f"Positive samples (true test edges): {sum(y_true)}")
    print(f"Negative samples: {len(y_true) - sum(y_true)}")
    print(f"Positive ratio: {sum(y_true)/len(y_true):.4f}")
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Node Prediction ROC - Adamic-Adar + Sequence\\n({sum(y_true)} pos / {len(y_true)} total samples, {sum(y_true)/len(y_true)*100:.1f}% positive)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('adamic_adar_node_roc.png', dpi=100)
    plt.close()
    print(f"ROC curve saved to adamic_adar_node_roc.png")
    
    return auc_score


def tune_k(G_full, embeddings, gene_to_protein_id, k_values=[1, 2, 3, 5, 10, 15, 20, 30], 
           val_ratio=0.2, test_ratio=0.2, seed=42, scoring_method='adamic_adar'):
    """
    Tune the k hyperparameter (number of virtual neighbors) using validation split.
    
    Args:
        G_full: full graph
        embeddings: protein embeddings
        gene_to_protein_id: mapping dict
        k_values: list of k values to try
        val_ratio: ratio of nodes for validation
        test_ratio: ratio of nodes for test
        seed: random seed
        
    Returns:
        best_k: optimal k value
    """
    print("\n--- Starting k Tuning ---")
    
    # Use utils.graph_split_rnd for consistent node splitting
    # First split: separate validation nodes
    G_val_train, val_test_edges, _ = utils.graph_split_rnd(
        G_full, test_ratio=val_ratio, split_type='vertex', rnd_seed=seed
    )
    
    # Extract validation nodes (those with degree 0 after split)
    val_nodes = [node for node in G_full.nodes() if G_val_train.degree(node) == 0]
    
    # Rebuild G_val_train to only contain nodes with edges
    G_val_train = G_val_train.subgraph([n for n in G_val_train.nodes() if G_val_train.degree(n) > 0]).copy()
    
    print(f"Train nodes: {len(G_val_train.nodes())}, Val nodes: {len(val_nodes)}")
    print(f"Validation test edges: {len(val_test_edges)}")
    
    best_k = None
    best_f1 = -1
    k_results = {}
    
    for k in k_values:
        print(f"\nTrying k={k}...")
        model = AdamicAdarSequence(G_val_train, embeddings, gene_to_protein_id, k=k, scoring_method=scoring_method)
        
        # Evaluate on validation nodes using val_test_edges
        accuracy, precision, recall, f1, tp, fp, tn, fn = node_level_evaluation_with_edges(model, val_nodes, val_test_edges, top_k=10)
        
        k_results[k] = f1
        print(f"k={k}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
    
    print(f"\nBest k found: {best_k} with F1: {best_f1:.4f}")
    
    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_results.keys()), list(k_results.values()), marker='o', color='teal')
    plt.title("k Optimization: Validation F1 Score")
    plt.xlabel("k (Number of Virtual Neighbors)")
    plt.ylabel("F1 Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('adamic_adar_k_tuning.png')
    plt.close()  # Close instead of show to avoid blocking
    print("K-tuning plot saved to adamic_adar_k_tuning.png")
    
    return best_k


def main():
    """
    Main execution function for Adamic-Adar + Sequence Features prediction.
    """
    parser = argparse.ArgumentParser(
        description='Node prediction using sequence features with configurable scoring method'
    )
    parser.add_argument(
        '--scoring-method',
        choices=['adamic_adar', 'common_neighbors'],
        default='adamic_adar',
        help='Scoring method to use (default: adamic_adar)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=None,
        help='Number of virtual neighbors (if not specified, will tune automatically)'
    )
    parser.add_argument(
        '--graph-file',
        default='string_interaction_physical.tsv',
        help='Path to graph file (default: string_interaction_physical.tsv)'
    )
    parser.add_argument(
        '--embeddings-file',
        default='protein_embeddings.pkl',
        help='Path to embeddings file (default: protein_embeddings.pkl)'
    )
    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip k tuning (requires --k to be specified)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_tuning and args.k is None:
        parser.error('--skip-tuning requires --k to be specified')
    
    print("=== Adamic-Adar with Sequence Features for Node Prediction ===\n")
    
    # Load data
    print("Loading graph...")
    G_full = utils.load_graph(args.graph_file)
    
    print("\nLoading protein embeddings...")
    embeddings = utils.load_protein_embeddings(args.embeddings_file)
    
    print("\nCreating gene name to protein ID mapping...")
    gene_to_protein_id = utils.create_protein_id_mapping(args.graph_file)
    
    # Determine optimal k
    if args.k is not None and args.skip_tuning:
        optimal_k = args.k
        print(f"\nUsing k={optimal_k} (skipping tuning)")
    elif args.k is not None:
        print(f"\n--- Tuning k hyperparameter around k={args.k} (using {args.scoring_method}) ---")
        k_range = [max(1, args.k-2), max(1, args.k-1), args.k, args.k+1, args.k+2]
        optimal_k = tune_k(
            G_full, embeddings, gene_to_protein_id,
            k_values=k_range,
            val_ratio=0.15,
            test_ratio=0.2,
            seed=42,
            scoring_method=args.scoring_method
        )
    else:
        # Tune k on a subset
        print(f"\n--- Tuning k hyperparameter (using {args.scoring_method}) ---")
        optimal_k = tune_k(
            G_full, embeddings, gene_to_protein_id,
            k_values=[1, 2, 3, 5, 10, 15, 20, 30],
            val_ratio=0.15,
            test_ratio=0.2,
            seed=42,
            scoring_method=args.scoring_method
        )
    
    # Split nodes for train/test using utils.graph_split_rnd
    print("\n--- Performing Node-Level Split ---")
    G_train, test_edges, _ = utils.graph_split_rnd(
        G_full, test_ratio=0.2, split_type='vertex', rnd_seed=42
    )
    
    # Extract test node names from test_edges
    # For vertex split, graph_split_rnd removes edges touching selected nodes
    # The selected test nodes are those with 0 degree in G_train (all edges removed)
    test_nodes = [node for node in G_full.nodes() if G_train.degree(node) == 0]
    
    # Rebuild G_train to only contain nodes with edges (remove isolated test nodes)
    G_train = G_train.subgraph([n for n in G_train.nodes() if G_train.degree(n) > 0]).copy()
    
    print(f"Total nodes: {G_full.number_of_nodes()}")
    print(f"Training nodes: {len(G_train.nodes())}")
    print(f"Test nodes (fully isolated): {len(test_nodes)}")
    print(f"Training edges: {G_train.number_of_edges()}")
    
    # Train final model with optimal k
    print(f"\n--- Training final model with k={optimal_k} and {args.scoring_method} ---")
    model = AdamicAdarSequence(G_train, embeddings, gene_to_protein_id, k=optimal_k, scoring_method=args.scoring_method)
    
    # Evaluate on test nodes using the removed test_edges
    print(f"\n--- Evaluating on {len(test_nodes)} test nodes ---")
    print(f"Test edges to predict: {len(test_edges)}")
    accuracy, precision, recall, f1, tp, fp, tn, fn = node_level_evaluation_with_edges(model, test_nodes, test_edges, top_k=10)
    
    print(f"\n{args.scoring_method.upper().replace('_', ' ')} + Sequence Features Performance (Top-10):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    
    # Compute AUC using the removed test_edges
    auc_score = compute_auc_for_nodes(model, test_nodes, test_edges)
    if auc_score:
        print(f"\nAUC Score: {auc_score:.4f}")
    
    # Show sample predictions
    print(f"\n--- Sample Predictions (First 5 Test Nodes) ---")
    sample_predictions = model.predict_for_multiple_nodes(test_nodes[:5], top_k=10)
    
    for test_node, predicted_partners in sample_predictions.items():
        actual_neighbors = set(G_full.neighbors(test_node)) if test_node in G_full else set()
        print(f"\nTest Node: {test_node}")
        print(f"Actual neighbors in full graph: {len(actual_neighbors)}")
        print("Top 10 Predicted Partners:")
        
        for i, (partner, score) in enumerate(predicted_partners, 1):
            is_correct = "✓" if partner in actual_neighbors else "✗"
            print(f"  {i}. {partner:<15} (score: {score:.4f}) {is_correct}")


if __name__ == '__main__':
    main()
