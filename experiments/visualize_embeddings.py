"""
Visualize node embeddings using t-SNE for Baseline, SynFlow, and Random pruning.

This script:
1. Trains temporary models for:
   - Baseline (Unpruned)
   - SynFlow (95% Sparsity)
   - Random (95% Sparsity)
2. Extracts embeddings from each.
3. Projects them to 2D using t-SNE.
4. Generates side-by-side plots to visualize the quality of the learned latent space.

Usage:
    python experiments/visualize_embeddings.py
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.dgraph_fin import load_dgraphfin_dataset
from src.models.graphsage import build_graphsage_for_data
from src.training.train_baseline import build_loaders, compute_pos_weight
from src.pruning.synflow import prune_model_synflow, enforce_masks
from src.pruning.random import prune_model_random

def train_demo_model(dataset, variant, sparsity, device, epochs=5):
    """
    Trains a demo model (Baseline, SynFlow, or Random) for a few epochs.
    """
    graph = dataset[0]
    graph = graph.to(device)
    
    print(f"\n--- Training Demo Model: {variant} (Sparsity={sparsity}) ---")
    
    # Config
    hidden_channels = 128
    num_layers = 2
    dropout = 0.2
    lr = 0.01
    
    model = build_graphsage_for_data(
        graph,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    # Apply Pruning BEFORE training (Pruning at Initialization)
    if variant == "SynFlow":
        model = prune_model_synflow(model, graph, sparsity)
    elif variant == "Random":
        model = prune_model_random(model, sparsity)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(graph.ndata['label']).to(device))
    
    # Loaders (Use Heuristic for Pruned, Uniform for Baseline to match experiments)
    class DemoConfig:
        sampling = "heuristic" if variant != "Baseline" else "uniform"
        batch_size = 4096
        num_neighbors = [10, 10]
        num_hops = 2
        k_pos = 10
        k_neg = 10
        
    print(f"Sampling: {DemoConfig.sampling}")
    train_loader = build_loaders(graph.cpu(), DemoConfig(), device)
    
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_item in pbar:
            optimizer.zero_grad()

            if DemoConfig.sampling == "uniform":
                _, _, blocks = batch_item
                blocks = [b.to(device) for b in blocks]
                input_feats = blocks[0].srcdata['feat']
                output_labels = blocks[-1].dstdata['label'].float()
                logits = model(blocks, input_feats).view(-1)
            else:
                # Heuristic sampling returns a SubgraphBatch
                batch = batch_item
                subgraph = batch.graph.to(device)
                feats = subgraph.ndata['feat']
                logits_all = model(subgraph, feats).view(-1)
                seed_idx = batch.seed_idx.to(device)
                logits = logits_all[seed_idx]
                output_labels = subgraph.ndata['label'][seed_idx].float()

            loss = criterion(logits, output_labels)
            loss.backward()
            optimizer.step()
            
            # Enforce masks if pruned
            if variant in ["SynFlow", "Random"]:
                enforce_masks(model)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    return model

def extract_embeddings(model, graph, device):
    """
    Extracts full-batch embeddings.
    """
    model.eval()
    model = model.to('cpu')
    graph = graph.to('cpu')
    
    with torch.no_grad():
        h = graph.ndata['feat']
        for conv in model.convs:
            h = conv(graph, h)
            h = model.activation(h)
        
        embeddings = h
        logits = model.out_proj(h).squeeze()
        probs = torch.sigmoid(logits)
        
    return embeddings, probs

def plot_tsne(embeddings, labels, probs, title, save_path):
    """
    Runs t-SNE and plots the result.
    """
    # Sample subset for visualization
    fraud_mask = (labels == 1)
    fraud_indices = fraud_mask.nonzero(as_tuple=True)[0]
    
    normal_mask = (labels == 0)
    normal_indices = normal_mask.nonzero(as_tuple=True)[0]
    
    n_fraud = len(fraud_indices)
    n_normal = n_fraud * 2 # 1:2 ratio
    
    perm_normal = torch.randperm(len(normal_indices))[:n_normal]
    sampled_normal = normal_indices[perm_normal]
    
    sampled_indices = torch.cat([fraud_indices, sampled_normal])
    
    X = embeddings[sampled_indices].numpy()
    y = labels[sampled_indices].numpy()
    p = probs[sampled_indices].numpy()
    
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', n_jobs=-1)
    X_emb = tsne.fit_transform(X)
    
    # Define Status
    preds = (p > 0.5).astype(int)
    status = []
    for true, pred in zip(y, preds):
        if true == 1 and pred == 1:
            status.append("Detected Fraud")
        elif true == 1 and pred == 0:
            status.append("Missed Fraud")
        elif true == 0 and pred == 0:
            status.append("True Normal")
        else:
            status.append("False Alarm")
    
    df = pd.DataFrame({
        "x": X_emb[:, 0],
        "y": X_emb[:, 1],
        "Status": status,
        "Label": ["Fraud" if l==1 else "Normal" for l in y]
    })
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        data=df, 
        x="x", y="y", 
        hue="Status", 
        palette={
            "Detected Fraud": "#C44E52", # Red
            "Missed Fraud": "#FFA07A",   # Orange
            "True Normal": "#4C72B0",    # Blue
            "False Alarm": "#8172B3"     # Purple
        },
        alpha=0.6,
        s=20,
        legend=False
    )
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = load_dgraphfin_dataset(root=project_root / "data" / "DGraphFin2")
    labels = dataset[0].ndata['label']
    
    save_dir = project_root / "results" / "plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Baseline
    model_base = train_demo_model(dataset, "Baseline", 0.0, device)
    emb_base, probs_base = extract_embeddings(model_base, dataset[0], device)
    plot_tsne(emb_base, labels, probs_base, "Baseline (Dense)", save_dir / "embedding_baseline.png")
    
    # 2. SynFlow (95%)
    model_syn = train_demo_model(dataset, "SynFlow", 0.95, device)
    emb_syn, probs_syn = extract_embeddings(model_syn, dataset[0], device)
    plot_tsne(emb_syn, labels, probs_syn, "SynFlow (95% Sparse)", save_dir / "embedding_synflow.png")
    
    # 3. Random (95%)
    model_rnd = train_demo_model(dataset, "Random", 0.95, device)
    emb_rnd, probs_rnd = extract_embeddings(model_rnd, dataset[0], device)
    plot_tsne(emb_rnd, labels, probs_rnd, "Random (95% Sparse)", save_dir / "embedding_random.png")
    
    print("\nDone! Check results/plots/ for the comparisons.")

if __name__ == "__main__":
    main()
