# Poisson Variational Autoencoder (P-VAE)

**Official PyTorch Implementation** | [NeurIPS 2024 Spotlight Paper](https://openreview.net/forum?id=ektPEcqGLb)

Welcome to the *Poisson Variational Autoencoder* (P-VAE) codebase! P-VAE is a brain-inspired generative model that unifies major theories in neuroscience with modern machine learning.

![Graphical Abstract](./media/graphical-abstract.png)

When trained on whitened natural image patches, the P-VAE learns sparse, "Gabor-like" features—remarkably similar to what's observed when recording from actual neurons in the primary visual cortex.

![Animation](./media/animation.gif)

**Learn more:**
- 📄 [Research paper](https://openreview.net/forum?id=ektPEcqGLb)
- 🐦 [X summary thread](https://x.com/hadivafaii/status/1794467115510227442)
- 🎥 [Talk](https://www.youtube.com/live/Y9hP79tBXHo)

---

## 📁 Project Structure

```
├── train.py              # 🚀 Unified training launcher (start here!)
├── configs/              # YAML configuration files
│   ├── poisson_default.yaml
│   ├── poisson_exact.yaml
│   ├── poisson_score.yaml
│   └── gumbel_softmax_default.yaml
├── main/                 # Core model and training code
│   ├── vae.py           # VAE architectures (Poisson, GumbelSoftmax, etc.)
│   ├── train_vae.py     # Training loop and utilities
│   └── config_vae.py    # Configuration classes
├── base/                 # Base classes and utilities
│   ├── distributions.py # Poisson reparameterization algorithms
│   ├── train_base.py    # Base trainer class
│   └── dataset.py       # Dataset loading utilities
├── analysis/             # Evaluation and analysis tools
├── scripts/              # Sweep runner and legacy scripts
├── checkpoints/          # Model checkpoints
├── figures/              # Generated visualizations
└── utils/                # Helper utilities
```

---

## 🚀 Quick Start

### Option 1: Train with YAML Config (Recommended)

```bash
# Train Poisson VAE with default settings
python train.py --config configs/poisson_default.yaml --device 0

# Train with exact loss computation
python train.py --config configs/poisson_exact.yaml --device 0

# Train GumbelSoftmax variant
python train.py --config configs/gumbel_softmax_default.yaml --device 0
```

### Option 2: Train with CLI Arguments

```bash
# Basic Poisson VAE
python train.py --device 0 --model poisson --dataset vH16 --seed 0

# With specific hyperparameters
python train.py --device 0 --model poisson --dataset vH16 \
    --epochs 3000 --temp_stop 0.1 --temp_anneal --seed 42

# GumbelSoftmax Poisson
python train.py --device 0 --model gumbel_poisson --dataset vH16 \
    --upperbound_method fixed --upperbound_param 50
```

### Option 3: Mix Config File with CLI Overrides

```bash
# Use config as base, override specific params
python train.py --config configs/poisson_default.yaml --seed 42 --epochs 1000
```

---

## ⚙️ Training Methods

The P-VAE supports three training methods:

| Method | Description | Use Case |
|--------|-------------|----------|
| `mc` | Monte Carlo sampling (default) | General training, soft reparameterization |
| `exact` | Analytical loss computation | Linear decoders, reduced gradient variance |
| `score` | Denoising score matching | Experimental, alternative gradient signal |

```bash
# Monte Carlo (default)
python train.py --device 0 --method mc ...

# Exact (analytical) - linear decoder only
python train.py --device 0 --method exact ...

# Score matching - experimental
python train.py --device 0 --method score ...
```

---

## 🔧 Configuration Reference

### Model Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `poisson` | Model type: `poisson`, `gumbel_poisson`, `gaussian`, `laplace`, `categorical` |
| `--dataset` | str | `vH16` | Dataset: `vH16`, `CIFAR16`, `MNIST`, `EMNIST`, `FashionMNIST` |
| `--architecture` | str | `lin\|lin` | Encoder\|Decoder: `lin\|lin`, `conv\|lin`, `mlp\|lin` |
| `--n_latents` | int | `512` | Latent dimensionality |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epochs` | int | `3000` | Number of training epochs |
| `--batch_size` | int | `64` | Batch size |
| `--lr` | float | `0.0002` | Learning rate |
| `--seed` | int | `0` | Random seed |

### Temperature Annealing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temp_start` | float | `1.0` | Starting temperature |
| `--temp_stop` | float | `0.1` | Final temperature |
| `--temp_anneal` | flag | `False` | Enable temperature annealing |
| `--temp_anneal_portion` | float | `0.5` | Portion of training for annealing |

### Poisson-Specific

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--indicator_approx` | str | `sigmoid` | Soft indicator: `sigmoid`, `cubic`, `linear`, `cosine` |
| `--prior_clamp` | float | `-4.0` | Prior log rate clamping |

### GumbelSoftmax-Specific

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--upperbound_method` | str | `fixed` | Method: `fixed`, `std_ratio`, `quantile`, `adaptive` |
| `--upperbound_param` | int | `50` | Upperbound parameter value |

### Logging

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no_wandb` | flag | `False` | Disable Weights & Biases logging |
| `--wandb_project` | str | `PoissonVAE` | W&B project name |
| `--wandb_entity` | str | `poisson-collab` | W&B entity |
| `--verbose` | flag | `False` | Enable verbose output |

---

## 📊 Running Sweeps

For systematic hyperparameter sweeps:

```bash
# Generate sweep queue
./scripts/run_sweep.sh generate

# Start sweep runner (runs in tmux background)
./scripts/run_sweep.sh run

# Check status
./scripts/run_sweep.sh status
```

---

## 📦 Checkpoints and Data

### Pre-trained Checkpoints

Linear VAE checkpoints are in `./checkpoints/`. To visualize:

```python
from main.vae import PoissonVAE
from base.utils_model import load_model_lite

model, cfg = load_model_lite('checkpoints/sweep/exp_sigmoid_t01_ann_seed0')
```

### Datasets

Download from [Google Drive](https://drive.google.com/drive/folders/1mCrsYtxcbNODcCTCLdaTi5v8yN_n5AMA?usp=sharing) and place in `~/Datasets/`:

```
~/Datasets/
├── DOVES/vH16/          # Van Hateren natural images
├── CIFAR16/xtract16/    # CIFAR 16x16 patches  
└── MNIST/processed/     # MNIST digits
```

---

## 🔬 PyTorch Lightning Implementation

For a minimal standalone implementation:

<a target="_blank" href="https://colab.research.google.com/drive/1PBeAv-3kcrwrSBKzRxDCcDiaxyC0-8Xv?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## 📚 Citation

```bibtex
@inproceedings{vafaii2024poisson,
    title={Poisson Variational Autoencoder},
    author={Hadi Vafaii and Dekel Galor and Jacob L. Yates},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=ektPEcqGLb},
}
```

---

## 📬 Contact

- **Code issues**: Open an issue in this repository
- **Paper questions**: [vafaii@berkeley.edu](mailto:vafaii@berkeley.edu)
