# Virtual Environment Setup

This project uses a Python virtual environment to manage dependencies.

## Activating the Virtual Environment

To activate the virtual environment, run:

```bash
source venv/bin/activate
```

Or if you're in the project directory:

```bash
cd /Users/jameshall/UCLA/260D/Project
source venv/bin/activate
```

Once activated, you should see `(venv)` at the beginning of your terminal prompt.

## Deactivating the Virtual Environment

To deactivate the virtual environment:

```bash
deactivate
```

## Installing Additional Packages

If you need to install additional packages:

```bash
# Make sure the venv is activated first
pip install <package-name>

# Then update requirements.txt
pip freeze > requirements.txt
```

## Key Libraries Installed

- **PyTorch** - Deep learning framework
- **DGL** - Graph neural network library (GraphSAGE, sampling, etc.)
- **NumPy, Pandas, SciPy** - Data processing
- **scikit-learn** - Machine learning utilities
- **torch-pruning** - Model pruning utilities
- **imbalanced-learn** - Handling imbalanced datasets
- **Matplotlib, Seaborn** - Visualization
- **Jupyter** - Notebook support
- **TensorBoard** - Experiment tracking

## Notes

- The virtual environment is located in the `venv/` directory
- Do not commit the `venv/` directory to version control (add it to `.gitignore`)
- Always activate the virtual environment before running project code

---

## Environment Setup Note for macOS (Apple Silicon/M1/M2) and Colab

This project now uses **DGL** for graph neural network layers and sampling. Installation is much simpler than the previous PyG setup.

### macOS (Apple Silicon/M1/M2)

1. Create/activate the virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

2. Install dependencies (CPU-only, works on M1/M2):

```bash
pip install -r requirements.txt
```

This will install `torch==2.0.0` and `dgl` with CPU support, which is sufficient for development and smaller runs.

### Google Colab

In a Colab notebook cell, you can install the dependencies with:

```python
!pip install torch==2.0.0 torchvision==0.15.1 torchaudio==0.15.1
!pip install dgl==2.4.0
!pip install -r requirements.txt
```

If you want a GPU-accelerated DGL build, you can instead install the CUDA-specific DGL wheel following the [official DGL installation instructions](https://www.dgl.ai/pages/start.html) for the CUDA version used by Colab, then install the rest of the requirements.


