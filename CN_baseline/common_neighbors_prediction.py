import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import argparse

def load_graph(file_path):
    """
    Load the protein interaction graph from TSV file.
    Returns:
        - edges: dict mapping (node1, node2) to combined_score
        - nodes: set of all unique nodes
        - adjacency: dict mapping each node to its neighbors with scores
    """
    # Read the file with proper header (line starting with # is the header)
    with open(file_path, 'r') as f:
        # Read lines to find the header
        lines = f.readlines()
    
    # Find header line (starts with #) and data lines
    header_line = None
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#'):
            header_line = line[1:].strip()  # Remove # and whitespace
            data_start = i + 1
            break
    
    if header_line is None:
        raise ValueError("No header line found (line starting with #)")
    
    # Read data with the correct header
    from io import StringIO
    data = ''.join(lines[data_start:])
    df = pd.read_csv(StringIO(header_line + '\n' + data), sep='\t')
    
    # Create edge dictionary and adjacency list
    edges = {}
    adjacency = defaultdict(dict)
    nodes = set()
    
    for _, row in df.iterrows():
        node1 = row['node1']
        node2 = row['node2']
        score = row['combined_score']
        
        # Add to edges (treat as undirected)
        edges[(node1, node2)] = score
        edges[(node2, node1)] = score
        
        # Add to adjacency list
        adjacency[node1][node2] = score
        adjacency[node2][node1] = score
        
        # Track all nodes
        nodes.add(node1)
        nodes.add(node2)
    
    return edges, nodes, adjacency

def calculate_common_neighbor_score(node_x, node_y, adjacency):
    """
    Calculate the common neighbor score for two nodes x and y.
    For each common neighbor z, add min(score(z,x), score(z,y)) to the sum.
    
    Args:
        node_x: First node
        node_y: Second node
        adjacency: Dictionary mapping nodes to their neighbors and scores
    
    Returns:
        Common neighbor score (sum of minimum scores)
    """
    neighbors_x = set(adjacency[node_x].keys())
    neighbors_y = set(adjacency[node_y].keys())
    
    # Find common neighbors
    common_neighbors = neighbors_x & neighbors_y
    
    # Calculate score as sum of minimum combined_scores
    total_score = 0.0
    for z in common_neighbors:
        score_zx = adjacency[node_x][z]
        score_zy = adjacency[node_y][z]
        total_score += min(score_zx, score_zy)
    
    return total_score

def predict_edges(edges, nodes, adjacency, threshold=0.5, top_k=None):
    """
    Predict new edges based on common neighbor scores.
    
    Args:
        edges: Existing edges dictionary
        nodes: Set of all nodes
        adjacency: Adjacency list with scores
        threshold: Minimum score threshold for predicting an edge
        top_k: If specified, return only top K predictions
    
    Returns:
        List of tuples (node1, node2, score) for predicted edges
    """
    predictions = []
    node_list = list(nodes)
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Existing edges: {len(edges) // 2}")  # Divide by 2 because we store both directions
    
    # Consider all pairs of nodes that don't have an edge
    total_pairs = 0
    for i, node_x in enumerate(node_list):
        if i % 100 == 0:
            print(f"Processing node {i}/{len(node_list)}...")
        
        for node_y in node_list[i+1:]:
            total_pairs += 1
            # Check if edge doesn't exist
            if (node_x, node_y) not in edges:
                score = calculate_common_neighbor_score(node_x, node_y, adjacency)
                
                if score > 0:  # Only consider pairs with at least one common neighbor
                    predictions.append((node_x, node_y, score))
    
    print(f"Total node pairs evaluated: {total_pairs}")
    print(f"Pairs with common neighbors: {len(predictions)}")
    
    # Sort by score (descending)
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    # Filter by threshold
    filtered_predictions = [(n1, n2, s) for n1, n2, s in predictions if s >= threshold]
    
    print(f"Predictions above threshold {threshold}: {len(filtered_predictions)}")
    
    # Return top K if specified
    if top_k is not None:
        return filtered_predictions[:top_k]
    
    return filtered_predictions

def save_predictions(predictions, output_file):
    """Save predictions to a TSV file."""
    df = pd.DataFrame(predictions, columns=['node1', 'node2', 'common_neighbor_score'])
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Predictions saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Predict edges using Common Neighbors baseline model'
    )
    parser.add_argument(
        'input_file',
        help='Input TSV file with protein interactions'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Minimum common neighbor score threshold (default: 0.5)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Return only top K predictions (default: all above threshold)'
    )
    parser.add_argument(
        '--output',
        default='predicted_edges.tsv',
        help='Output file for predictions (default: predicted_edges.tsv)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading graph from {args.input_file}...")
    edges, nodes, adjacency = load_graph(args.input_file)
    
    print(f"\nPredicting edges with Common Neighbors model...")
    print(f"Threshold: {args.threshold}")
    if args.top_k:
        print(f"Top K: {args.top_k}")
    
    predictions = predict_edges(edges, nodes, adjacency, args.threshold, args.top_k)
    
    if predictions:
        print(f"\n=== Top 10 Predictions ===")
        for node1, node2, score in predictions[:10]:
            print(f"{node1} <-> {node2}: {score:.3f}")
        
        save_predictions(predictions, args.output)
        
        # Print statistics
        scores = [s for _, _, s in predictions]
        print(f"\n=== Statistics ===")
        print(f"Total predictions: {len(predictions)}")
        print(f"Score range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"Mean score: {np.mean(scores):.3f}")
        print(f"Median score: {np.median(scores):.3f}")
    else:
        print("\nNo predictions found above the threshold.")

if __name__ == '__main__':
    main()
