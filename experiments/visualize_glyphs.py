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

def plot_ego_graph(graph, center_node_idx, label_map, title, save_path, degrees=None):
    """
    Extracts and plots the 2-hop ego network for a center node.
    """
    # 1. Extract k-hop subgraph (k=2)
    sg, inverse_indices = dgl.khop_in_subgraph(graph, center_node_idx, k=2)
    
    # Get global IDs of nodes in the subgraph
    # sg.ndata[dgl.NID] stores the original node IDs
    global_ids = sg.ndata[dgl.NID]
    
    # Prepare node attributes
    
    # Node Degree for Size
    if degrees is not None:
        subgraph_degrees = degrees[global_ids].float()
        # Log scale for size: base + multiplier * log(deg + 1)
        # Clip to avoid tiny or massive nodes
        sizes = (torch.log1p(subgraph_degrees) * 100).numpy()
        sizes = np.clip(sizes, 100, 1000)
        sg.ndata['size'] = torch.from_numpy(sizes)
    else:
        sg.ndata['size'] = torch.ones(sg.num_nodes()) * 100

    # Edge Timestamp for Width
    if 'timestamp' in graph.edata:
        # Map subgraph edges to original edges
        orig_eids = sg.edata[dgl.EID]
        timestamps = graph.edata['timestamp'][orig_eids].float()
        
        # Normalize timestamps to [0.5, 4.0]
        if timestamps.numel() > 0:
            t_min, t_max = timestamps.min(), timestamps.max()
            if t_max > t_min:
                # Normalize to 0-1
                norm_ts = (timestamps - t_min) / (t_max - t_min)
                # Map to width
                widths = 0.5 + 3.5 * norm_ts
            else:
                widths = torch.ones_like(timestamps) * 2.0
        else:
             widths = torch.tensor([])
             
        sg.edata['width'] = widths
    else:
        # Default width if no timestamp
        sg.edata['width'] = torch.ones(sg.num_edges()) * 1.0

    # Convert to NetworkX, carrying over attributes
    # node_attrs=['label', 'size'], edge_attrs=['width']
    # Note: dgl.to_networkx copies features to the nx graph
    # sg.ndata['label'] is automatically inherited/copied by dgl.khop_in_subgraph
    nx_g = dgl.to_networkx(sg, node_attrs=['label', 'size'], edge_attrs=['width'])
    
    node_colors = []
    node_sizes = []
    
    # Find local ID of center node
    # inverse_indices returned by khop_in_subgraph contains the LOCAL IDs of the seed nodes
    # (not the global IDs as previously assumed)
    if isinstance(inverse_indices, torch.Tensor):
        center_node_local_id = inverse_indices.item()
    else:
        center_node_local_id = inverse_indices

    # Extract node attributes from nx_g
    nodelist = list(nx_g.nodes())
    
    for n in nodelist:
        # Attributes
        attrs = nx_g.nodes[n]
        label = attrs['label'].item()
        size = attrs['size'].item()
        
        # Color
        color = "#C44E52" if label == 1 else "#4C72B0"
        node_colors.append(color)
        
        # Highlight center node size slightly more? 
        # Or just rely on its natural degree (which should be high if it's a hub, low otherwise)
        # The prompt says "Glyph: Use node size for degree".
        # But we might want to ensure the center is visible.
        # Let's keep the degree-based size, but maybe ensure a minimum for center?
        if n == center_node_local_id:
             size = max(size, 300) # Ensure center is at least visible
        
        node_sizes.append(size)

    # Edge widths
    edge_widths = []
    edgelist = list(nx_g.edges())
    for u, v in edgelist:
        # For MultiDiGraph, there's a key, but DGL default is DiGraph or MultiDiGraph?
        # DGL->NX usually MultiDiGraph? Let's check defaults. 
        # dgl.to_networkx returns MultiDiGraph if multigraph=True. Default is True?
        # Default is True.
        # If MultiDiGraph, edges keys are needed.
        # But if we iterate G.edges(data=True), we get attributes.
        pass

    # Re-iterate to get widths
    edge_widths = []
    # nx_g.edges(data=True) yields (u, v, d)
    for u, v, d in nx_g.edges(data=True):
        edge_widths.append(d.get('width', 1.0).item())

    # Layout
    pos = nx.spring_layout(nx_g, seed=42, k=0.15)
    
    plt.figure(figsize=(10, 10))
    
    # Draw edges
    nx.draw_networkx_edges(
        nx_g, 
        pos, 
        alpha=0.4, 
        edge_color='gray',
        width=edge_widths,
        node_size=node_sizes # Arrowheads need to know node size to stop early
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        nx_g, 
        pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        alpha=0.9,
        linewidths=1.5,
        edgecolors='white'
    )
    
    # Highlight center node specifically (border)
    if center_node_local_id != -1:
        nx.draw_networkx_nodes(
            nx_g, 
            pos, 
            nodelist=[center_node_local_id],
            node_size=node_sizes[nodelist.index(center_node_local_id)], 
            node_color=node_colors[nodelist.index(center_node_local_id)], 
            linewidths=3.0,
            edgecolors='gold'
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Normal User',
               markerfacecolor='#4C72B0', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Fraudster',
               markerfacecolor='#C44E52', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='Center Node',
               markerfacecolor='gray', markeredgecolor='gold', markersize=15, mew=2),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(f"{title}\n(Nodes: {sg.num_nodes()}, Edges: {sg.num_edges()})", fontsize=14)
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
                   project_root / "results" / "glyphs" / "random_fraud.png", degrees)
    
    # 2. Random Normal
    rand_normal = normal_indices[torch.randint(0, len(normal_indices), (1,)).item()]
    plot_ego_graph(graph, rand_normal, labels, "Random Normal User (2-hop)", 
                   project_root / "results" / "glyphs" / "random_normal.png", degrees)
    
    # 3. High-Degree Fraudster (Hub)
    # Filter fraud indices by degree
    fraud_degrees = degrees[fraud_indices]
    # Get top 5 highest degree fraudsters
    top_fraud_vals, top_fraud_args = torch.topk(fraud_degrees, k=5)
    hub_fraud = fraud_indices[top_fraud_args[0]].item()
    plot_ego_graph(graph, hub_fraud, labels, f"High-Degree Fraudster (Deg={top_fraud_vals[0].item()})", 
                   project_root / "results" / "glyphs" / "hub_fraud.png", degrees)

    # 4. High-Degree Normal (Hub)
    normal_degrees = degrees[normal_indices]
    top_normal_vals, top_normal_args = torch.topk(normal_degrees, k=5)
    hub_normal = normal_indices[top_normal_args[0]].item()
    plot_ego_graph(graph, hub_normal, labels, f"High-Degree Normal (Deg={top_normal_vals[0].item()})", 
                   project_root / "results" / "glyphs" / "hub_normal.png", degrees)

if __name__ == "__main__":
    main()

