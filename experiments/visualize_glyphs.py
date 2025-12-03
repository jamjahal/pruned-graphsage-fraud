"""
Visualize ego-network "glyphs" for specific nodes in the DGraph-Fin dataset.

This script:
1. Loads the DGraph-Fin dataset.
2. Selects a few interesting nodes:
   - Random Normal User
   - Random Fraudster
   - High-degree Normal
   - High-degree Fraudster
3. Extracts their k-hop subgraphs (k=2).
4. Visualizes them using NetworkX and Matplotlib, coloring nodes by label.
5. Saves the figures to `results/glyphs/`.

Usage:
    python experiments/visualize_glyphs.py
"""

import os
import sys
import random
from pathlib import Path

import dgl
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.dgraph_fin import load_dgraphfin_dataset

def plot_ego_graph(graph, center_node_idx, label_map, title, save_path):
    """
    Extracts and plots the 2-hop ego network for a center node.
    """
    # 1. Extract k-hop subgraph (k=2)
    # dgl.khop_in_subgraph returns the subgraph induced by the k-hop neighbors
    # AND the original node IDs (as induced_nodes)
    sg, inverse_indices = dgl.khop_in_subgraph(graph, center_node_idx, k=2)
    
    # Convert to NetworkX for plotting
    # The subgraph 'sg' has new node IDs (0 to N_subgraph).
    # We need to map labels from the original graph to these new IDs.
    # inverse_indices contains the original node IDs.
    nx_g = dgl.to_networkx(sg)
    
    # Get labels for nodes in subgraph
    # original_ids = inverse_indices.numpy()
    subgraph_labels = graph.ndata['label'][inverse_indices].numpy()
    
    # Prepare colors
    # 0 (Normal) -> Blue (#4C72B0)
    # 1 (Fraud)  -> Red  (#C44E52)
    node_colors = []
    node_sizes = []
    
    center_node_local_id = -1
    
    # Find the local ID of the center node. 
    # inverse_indices maps local_id -> global_id.
    # We want to find local_id where global_id == center_node_idx.
    try:
        # This works if center_node_idx is a scalar
        center_matches = (inverse_indices == center_node_idx).nonzero(as_tuple=True)[0]
        if len(center_matches) > 0:
            center_node_local_id = center_matches[0].item()
    except:
        pass

    for i, label in enumerate(subgraph_labels):
        if i == center_node_local_id:
            # Center node is distinct (larger, yellow border?)
            node_sizes.append(300)
            # Color by label still
            color = "#C44E52" if label == 1 else "#4C72B0"
            node_colors.append(color)
        else:
            node_sizes.append(50)
            color = "#C44E52" if label == 1 else "#4C72B0"
            node_colors.append(color)

    # Layout
    pos = nx.spring_layout(nx_g, seed=42, k=0.15)
    
    plt.figure(figsize=(8, 8))
    
    # Draw edges
    nx.draw_networkx_edges(nx_g, pos, alpha=0.3, edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(
        nx_g, 
        pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        alpha=0.9,
        linewidths=1.0,
        edgecolors='white'  # white border for all
    )
    
    # Highlight center node specifically
    if center_node_local_id != -1:
        nx.draw_networkx_nodes(
            nx_g, 
            pos, 
            nodelist=[center_node_local_id],
            node_size=300, 
            node_color=node_colors[center_node_local_id], 
            linewidths=2.0,
            edgecolors='gold'  # Gold border for center
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal User',
               markerfacecolor='#4C72B0', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Fraudster',
               markerfacecolor='#C44E52', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Center Node',
               markerfacecolor='gray', markeredgecolor='gold', markersize=10),
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"{title}\n(Nodes: {sg.num_nodes()}, Edges: {sg.num_edges()})")
    plt.axis('off')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved glyph to {save_path}")


def main():
    print("Loading DGraph-Fin dataset...")
    dataset = load_dgraphfin_dataset(root=project_root / "data" / "DGraphFin2")
    graph = dataset[0]
    labels = graph.ndata['label']
    degrees = graph.in_degrees()
    
    # Identify indices
    fraud_indices = (labels == 1).nonzero(as_tuple=True)[0]
    normal_indices = (labels == 0).nonzero(as_tuple=True)[0]
    
    print(f"Total Nodes: {graph.num_nodes()}")
    print(f"Fraud Nodes: {len(fraud_indices)}")
    print(f"Normal Nodes: {len(normal_indices)}")
    
    # 1. Random Fraudster
    rand_fraud = fraud_indices[torch.randint(0, len(fraud_indices), (1,)).item()]
    plot_ego_graph(graph, rand_fraud, labels, "Random Fraudster (2-hop)", 
                   project_root / "results" / "glyphs" / "random_fraud.png")
    
    # 2. Random Normal
    rand_normal = normal_indices[torch.randint(0, len(normal_indices), (1,)).item()]
    plot_ego_graph(graph, rand_normal, labels, "Random Normal User (2-hop)", 
                   project_root / "results" / "glyphs" / "random_normal.png")
    
    # 3. High-Degree Fraudster (Hub)
    # Filter fraud indices by degree
    fraud_degrees = degrees[fraud_indices]
    # Get top 5 highest degree fraudsters
    top_fraud_vals, top_fraud_args = torch.topk(fraud_degrees, k=5)
    hub_fraud = fraud_indices[top_fraud_args[0]].item()
    plot_ego_graph(graph, hub_fraud, labels, f"High-Degree Fraudster (Deg={top_fraud_vals[0].item()})", 
                   project_root / "results" / "glyphs" / "hub_fraud.png")

    # 4. High-Degree Normal (Hub)
    normal_degrees = degrees[normal_indices]
    top_normal_vals, top_normal_args = torch.topk(normal_degrees, k=5)
    hub_normal = normal_indices[top_normal_args[0]].item()
    plot_ego_graph(graph, hub_normal, labels, f"High-Degree Normal (Deg={top_normal_vals[0].item()})", 
                   project_root / "results" / "glyphs" / "hub_normal.png")

if __name__ == "__main__":
    main()

