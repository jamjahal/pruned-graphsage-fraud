"""
Script to update 'approx_flops' in existing seed_*.json files using the latest
implementation of estimate_graphsage_flops in src.metrics.efficiency.

Usage:
    python experiments/update_flops.py
"""

import json
import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.dgraph_fin import load_dgraphfin_dataset
from src.metrics.efficiency import estimate_graphsage_flops
from src.models.graphsage import build_graphsage_for_data
from src.pruning.magnitude import prune_model_magnitude
from src.pruning.synflow import prune_model_synflow

def update_flops_for_results():
    results_root = project_root / "results"
    
    # Load dataset once (needed for graph size)
    print("Loading dataset to get graph dimensions...")
    dataset = load_dgraphfin_dataset(root=project_root / "data" / "DGraphFin2")
    graph = dataset[0]
    num_nodes = graph.num_nodes()
    num_edges = graph.num_edges()
    print(f"Graph loaded: {num_nodes} nodes, {num_edges} edges.")

    # Iterate over all variant folders
    for variant_dir in sorted(results_root.glob("*")):
        if not variant_dir.is_dir():
            continue
        
        print(f"Processing variant: {variant_dir.name}")
        
        json_files = sorted(variant_dir.glob("seed_*.json"))
        if not json_files:
            continue

        # We can rebuild the model once per variant since architecture/sparsity 
        # is usually constant across seeds for the same variant.
        # However, to be safe and robust to potential config differences, 
        # let's read the config from the first seed and rebuild.
        
        first_seed_path = json_files[0]
        with open(first_seed_path, "r") as f:
            data = json.load(f)
        
        # Reconstruct minimal model to estimate FLOPs
        # We assume standard hyperparameters if not fully specified, 
        # but usually we can infer what we need.
        # Note: The JSON summary doesn't store full hyperparams (hidden_channels etc).
        # We will assume the defaults used in run_experiment or try to infer from config files.
        # LIMITATION: If experiments used different hidden_dims, we might need to read the config file.
        # But for this project, all variants use hidden_channels=128, num_layers=2.
        
        hidden_channels = 128
        num_layers = 2
        dropout = 0.2 # doesn't affect FLOPs
        
        model = build_graphsage_for_data(
            graph,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        
        model_variant = data.get("model_variant", "baseline")
        sparsity = data.get("sparsity", 0.0)
        
        # Apply pruning if needed to get correct sparsity mask for FLOP counting
        if "pruned" in model_variant and sparsity > 0:
            if model_variant == "pruned_magnitude":
                model = prune_model_magnitude(model, sparsity=sparsity)
            elif model_variant == "pruned_synflow":
                # SynFlow requires data for scoring
                model = prune_model_synflow(model, graph=graph, sparsity=sparsity)
        
        # Calculate FLOPs with current logic
        new_flops = estimate_graphsage_flops(model, num_nodes, num_edges)
        print(f"  > Recalculated FLOPs for {model_variant} (sparsity={sparsity}): {new_flops:.3e}")

        # Update all JSONs in this folder
        for json_path in json_files:
            with open(json_path, "r") as f:
                record = json.load(f)
            
            old_flops = record.get("approx_flops")
            record["approx_flops"] = new_flops
            
            with open(json_path, "w") as f:
                json.dump(record, f, indent=2)
                
            print(f"    Updated {json_path.name} (was {old_flops})")

if __name__ == "__main__":
    update_flops_for_results()

