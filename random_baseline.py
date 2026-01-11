import utils
import networkx as nx
import random
from itertools import combinations

class RandomBaseline:
    
    def __init__(self, G):
        self.G = G
        self.edge_prob = self._compute_edge_probability()
        
    def _compute_edge_probability(self):
        total_possible_edges = self.G.number_of_nodes() * (self.G.number_of_nodes() - 1) / 2
        actual_edges = self.G.number_of_edges()
        return actual_edges / total_possible_edges 
        
    def predict(self):
        # For each possible (non-existing) edge, predict based on edge probability
        predictions = {}
        nodes = list(self.G.nodes())
        for u, v in combinations(nodes, 2):  # Automatically all unique pairs
            if not self.G.has_edge(u, v):
                predictions[(u, v)] = random.random() < self.edge_prob
        return predictions
        
        
def compare_predictions(predictions, test_edges):
    tp = fp = tn = fn = 0
    for (u, v, weight) in test_edges:
        pred = predictions.get((u, v), predictions.get((v, u), False))
        actual = True  # Since these are test edges, they exist
        
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        elif not pred and actual:
            fn += 1
    
    return tp, fp, tn, fn
            
    
def calculate_metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1_score


def main():
    G = utils.load_graph()
    G_train, test_edges = utils.graph_train_test_split(G, test_ratio=0.2, rnd_seed=41)
    model = RandomBaseline(G_train)
    predictions = model.predict()
    tp, fp, tn, fn = compare_predictions(predictions, test_edges)
    
    # Print the number of edges guessed (correctly and incorrectly)
    num_predicted_edges = sum(predictions.values())
    print(f"Number of edges predicted to exist: {num_predicted_edges}/{len(predictions)}")
    
    accuracy, precision, recall, f1_score = calculate_metrics(tp, fp, tn, fn)
    
    print(f"Random Baseline Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    

if __name__ == '__main__':
    main()