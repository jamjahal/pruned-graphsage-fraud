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
- **PyTorch Geometric** - Graph neural network library
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

# Environment Setup Note for macOS (Apple Silicon/M1/M2)

This project requires PyTorch Geometric (PyG) C++ extensions (`torch-sparse`, `torch-scatter`, etc.). Installing these extensions on macOS Apple Silicon (M1/M2) with standard virtual environments frequently fails due to two main issues:

**Compiler Configuration**: The default macOS system compiler (clang) is incompatible with the PyG C++ source code.

**API Mismatch**: The isolated build environment often uses a version of a PyG extension (like `torch-scatter`) whose C++ APIs conflict with the installed PyTorch version (even when both are nominally compatible).

## ✅ Definitive Solution: Pinned Versions and Build Isolation Control

The only way to guarantee a successful installation is to use a specific, stable version set and force the build system to use the correct M1 compiler while disabling its isolation.

| Dependency | Pinned Version | Reason |
|------------|----------------|--------|
| PyTorch | 2.0.0 | Stable baseline. Downgraded from 2.3.0 to utilize a version set with more robust M1-compatible binary wheels. |
| PyG Extensions | Varies (e.g., `torch-scatter=2.0.9`) | Specific versions were chosen through trial-and-error to find the exact C++ API level compatible with `torch==2.0.0`. |
| Build Flag | `--no-build-isolation` | Critical fix. Forces the build process to directly access the active virtual environment, allowing it to find the installed torch module and avoid the `ModuleNotFoundError`. |

## 🚀 Implementation Steps

To successfully set up the environment, follow these steps in your project directory:

### 1. Update Dependencies

Ensure your `requirements.txt` file contains this exact, successful configuration:

```ini
# --- CORE FRAMEWORK (PINNED) ---

torch==2.0.0

torchvision==0.15.1

torchaudio==0.15.1



# --- PYG EXTENSIONS (SUCCESSFUL PINS) ---

torch-geometric==2.3.1

torch-sparse==0.6.17

torch-scatter==2.0.9 # The final successful version!

torch-cluster==1.6.1

torch-spline-conv==1.2.2



# ... include the rest of your non-PyG requirements here (numpy, pandas, etc.)
```

### 2. Reset Environment and Install Build Tools

Create a clean environment and ensure the Homebrew LLVM compiler is installed, as it is required for building C++ extensions on M1.

```bash
# Deactivate and reset environment
deactivate
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate

# Install necessary M1 build tools (if not already installed)
brew install cmake llvm libomp

# Set compiler environment variables for the current session
export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"
```

### 3. Install Packages (Disabling Build Isolation)

Run the final command to install all dependencies. The `--no-build-isolation` flag bypasses the error-prone default build process:

```bash
pip install -r requirements.txt \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html \
    --no-cache-dir \
    --no-build-isolation
```

### 4. Cleanup

After a successful installation, you should unset the temporary environment variables:

```bash
unset CC
unset CXX
unset LDFLAGS
unset CPPFLAGS
```

## 🔧 Troubleshooting

If you encounter issues:

1. **Always start fresh**: Delete and recreate the virtual environment if installation fails
2. **Check compiler paths**: Ensure Homebrew LLVM is properly installed with `brew install llvm`
3. **Verify environment variables**: Double-check that `CC` and `CXX` point to the correct clang binaries
4. **Use the exact versions**: Do not deviate from the pinned versions in `requirements.txt`
5. **No build isolation**: The `--no-build-isolation` flag is critical - don't skip it

## 📝 Notes for Apple Silicon Users

- This installation process has been tested and confirmed working on M1/M2 MacBooks
- The pinned versions provide stability but may not be the absolute latest
- Consider upgrading PyTorch versions only after testing that PyG extensions remain compatible
- The CPU wheel index (`-f https://data.pyg.org/whl/torch-2.0.0+cpu.html`) ensures you get pre-compiled binaries where possible

