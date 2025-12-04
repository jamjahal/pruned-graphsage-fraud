"""
Visualize layer-wise sparsity distributions for Magnitude vs. SynFlow pruning.

This script:
1. Loads the DGraph-Fin dataset.
2. Instantiates two identical GraphSAGE models.
3. Prunes one with Magnitude (at init) and one with SynFlow (at init).
4. Calculates the sparsity (percentage of zeros) for each weight matrix.
5. Plots a grouped bar chart comparing the two methods.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.dgraph_fin import load_dgraphfin_dataset
from src.models.graphsage import build_graphsage_for_data
from src.pruning.magnitude import prune_model_magnitude
from src.pruning.synflow import prune_model_synflow, enforce_masks

def get_model_sparsity(model):
    """
    Returns a dictionary mapping parameter names to their sparsity ratio.
    Sparsity = (count of zeros) / (total elements).
    """
    sparsity_dict = {}
    
    # Check for registered masks (buffers)
    # The pruning utils register masks as buffers named "{param_name}_mask"
    for name, buf in model.named_buffers():
        if name.endswith("_mask"):
            param_name = name.replace("_mask", "")
            
            # Calculate sparsity of the MASK itself
            # (1 means kept, 0 means pruned)
            # So sparsity = 1.0 - mean(mask)
            sparsity = 1.0 - buf.float().mean().item()
            
            # Clean up name for display
            # e.g. "convs.0.fc_neigh.weight" -> "L1 Neigh"
            clean_name = param_name.replace("convs.", "Layer ")
            
            sparsity_dict[clean_name] = sparsity
            
    return sparsity_dict

def main():
    print("Loading DGraph-Fin dataset...")
    dataset = load_dgraphfin_dataset(root=project_root / "data" / "DGraphFin2")
    graph = dataset[0]
    
    # Use CPU for analysis to avoid OOM on large graph if GPU is limited
    device = torch.device("cpu")
    graph = graph.to(device)
    
    # Configuration (matching pruned_synflow_95.yaml)
    sparsity_target = 0.95
    hidden_channels = 128
    num_layers = 2
    
    print(f"Initializing models (Target Sparsity: {sparsity_target})...")
    
    # --- Magnitude Pruning ---
    model_mag = build_graphsage_for_data(
        graph, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers
    ).to(device)
    
    print("Applying Magnitude pruning...")
    prune_model_magnitude(model_mag, sparsity=sparsity_target)
    mag_sparsity = get_model_sparsity(model_mag)
    
    # --- SynFlow Pruning ---
    model_syn = build_graphsage_for_data(
        graph, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers
    ).to(device)
    
    print("Applying SynFlow pruning (this involves a forward/backward pass)...")
    # SynFlow needs gradients
    model_syn.train() 
    prune_model_synflow(model_syn, graph=graph, sparsity=sparsity_target)
    syn_sparsity = get_model_sparsity(model_syn)
    
    # --- Prepare Data for Plotting ---
    data = []
    for name, sp in mag_sparsity.items():
        data.append({"Parameter": name, "Method": "Magnitude", "Sparsity": sp})
        
    for name, sp in syn_sparsity.items():
        data.append({"Parameter": name, "Method": "SynFlow", "Sparsity": sp})
        
    df = pd.DataFrame(data)
    
    print("\nSparsity Statistics:")
    print(df)
    
    # --- Plotting ---
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Create grouped bar chart
    ax = sns.barplot(
        data=df, 
        x="Parameter", 
        y="Sparsity", 
        hue="Method",
        palette={"Magnitude": "#4C72B0", "SynFlow": "#C44E52"}
    )
    
    # Add reference line for target sparsity
    plt.axhline(y=sparsity_target, color='black', linestyle='--', label='Target Global Sparsity')
    
    plt.title(f"Layer-wise Sparsity Distribution (Global Target: {sparsity_target:.0%})")
    plt.xlabel("Weight Matrix")
    plt.ylabel("Sparsity (Fraction of Zeros)")
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right')
    plt.xticks(rotation=45, ha='right')
    
    # Save
    save_dir = project_root / "results" / "plots"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir / "sparsity_heatmap.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")

if __name__ == "__main__":
    main()

