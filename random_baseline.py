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


def main():
    G = utils.load_graph()
    G_train, test_edges = utils.graph_train_test_split(G, test_ratio=0.2, rnd_seed=41)
    model = RandomBaseline(G_train)
    predictions = model.predict()
    tp, fp, tn, fn = utils.compare_predictions(predictions, test_edges)
    
    # Print the number of edges guessed (correctly and incorrectly)
    num_predicted_edges = sum(predictions.values())
    print(f"Number of edges predicted to exist: {num_predicted_edges}/{len(predictions)}")
    
    accuracy, precision, recall, f1_score = utils.calculate_metrics(tp, fp, tn, fn)
    
    print(f"Random Baseline Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    

if __name__ == '__main__':
    main()