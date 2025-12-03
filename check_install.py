
import torch
import dgl
import dgl.nn as dglnn

def verify_install():
    print(f"Torch version: {torch.__version__}")
    print(f"DGL version: {dgl.__version__}")
    
    # 1. Check Device (MPS is available for Torch, but DGL needs CPU)
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 2. Create a simple graph on CPU
    g = dgl.graph(([0, 1], [1, 2]))
    g = dgl.add_self_loop(g)
    feat = torch.randn(3, 5)
    
    # 3. Run a GraphConv layer
    conv = dglnn.GraphConv(5, 2, norm='both', weight=True, bias=True)
    res = conv(g, feat)
    
    print("\nSuccess! DGL GraphConv run on CPU produced:")
    print(res)
    print("\nConfiguration is ready for local M1 training.")

if __name__ == "__main__":
    verify_install()
