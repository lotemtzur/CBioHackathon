import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def analyze_graph_topology(edge_list_file):
    """
    Analyzes graph topology for STRING data.
    """
    print(f"--- Analysis Started for {edge_list_file} ---")

    # 1. Load Data
    try:
        # STRING files are tab-separated (.tsv)
        df = pd.read_csv(edge_list_file, sep='\t')

        # Print columns to verify headers
        print(f"Columns found: {list(df.columns)}")

        # STRING usually provides 'protein1', 'protein2', and 'combined_score'
        # Adjust these names if your file header is different
        source_col = 'protein1' if 'protein1' in df.columns else df.columns[0]
        target_col = 'protein2' if 'protein2' in df.columns else df.columns[1]
        score_col = 'combined_score' if 'combined_score' in df.columns else None

        if score_col:
            G = nx.from_pandas_edgelist(df, source=source_col, target=target_col, edge_attr=score_col)
        else:
            G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)

        print("✅ Data loaded successfully.")

    except FileNotFoundError:
        print(f"❌ Error: File '{edge_list_file}' not found. Please upload it.")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 2. Basic Stats
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    print(f"\n📊 Graph Statistics:")
    print(f"Nodes: {num_nodes}")
    print(f"Edges: {num_edges}")
    print(f"Density: {density:.5f}")

    # 3. Connected Components
    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_cc = len(max(components, key=len))

    print(f"\n🔗 Connectivity:")
    print(f"Connected Components: {num_components}")
    print(f"Largest Component: {largest_cc} nodes ({largest_cc / num_nodes:.1%} of graph)")

    if num_components > 1:
        print("⚠️ Note: Graph is fragmented. Standard GCNs only propagate within components.")

    # 4. Degree Distribution
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)

    print(f"\n🕸️ Degree Distribution:")
    print(f"Average Degree: {avg_degree:.2f}")
    print(f"Max Degree: {max_degree}")

    if max_degree > 5 * avg_degree:
        print("💡 Insight: High variance (Hubs detected). GAT might handle this better, but GCN is still a solid start.")
    else:
        print("💡 Insight: Uniform degrees. GCN is optimal.")

    # 5. Plotting
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(degrees, bins=50, color='skyblue', edgecolor='black')
    plt.title("Degree Distribution")
    plt.xlabel("Degree (k)")
    plt.ylabel("Count")
    plt.yscale('log')

    # Visualization (Subsample if large)
    plt.subplot(1, 2, 2)
    limit = 500 if num_nodes > 500 else num_nodes
    subgraph = G.subgraph(list(G.nodes)[:limit])
    plt.title(f"Graph Visualization (First {limit} nodes)")

    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(subgraph, pos, node_size=15, alpha=0.6, node_color='teal', with_labels=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Pointing to your specific file
    analyze_graph_topology("../string_interaction_physical.tsv")