#!/usr/bin/env python3
"""
Unified Training Launcher for Poisson VAE

This is the main entry point for training VAE models. It provides:
- YAML/JSON configuration file support
- Command-line argument override
- Automatic run naming and checkpoint organization
- Weights & Biases integration
- GPU selection and management

Usage:
    # Train with config file
    python train.py --config configs/poisson_default.yaml --device 0
    
    # Train with CLI arguments (no config file)
    python train.py --device 0 --model poisson --dataset vH16
    
    # Override config with CLI args
    python train.py --config configs/base.yaml --seed 42 --epochs 1000

Examples:
    # Poisson VAE with exponential reparameterization
    python train.py --device 0 --model poisson --dataset vH16 \\
        --indicator_approx sigmoid --temp_stop 0.1 --seed 0
    
    # GumbelSoftmax Poisson VAE
    python train.py --device 0 --model gumbel_poisson --dataset vH16 \\
        --upperbound_method fixed --upperbound_param 50 --temp_stop 0.1
    
    # With exact loss computation (linear decoder only)
    python train.py --device 0 --model poisson --method exact --dataset vH16

See configs/ directory for example configuration files.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file (.yaml, .yml, or .json)
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    with open(path, 'r') as f:
        if suffix in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML configs. Install: pip install pyyaml")
        elif suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .json")


def _save_config_to_dir(config_obj, save_dir: Path, name: str):
    """
    Save a config object to a directory as JSON.
    
    Args:
        config_obj: Config object to save
        save_dir: Directory to save to
        name: Name for the config file (without .json extension)
    """
    # Get all attributes from the config object
    attrs = {}
    for key in dir(config_obj):
        if key.startswith('_'):
            continue
        value = getattr(config_obj, key)
        if callable(value):
            continue
        # Only save serializable types
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            attrs[key] = value
    
    # Save as JSON
    file_path = save_dir / f"{name}.json"
    with open(file_path, 'w') as f:
        json.dump(attrs, f, indent=2)


def generate_run_name(
    model_type: str,
    dataset: str,
    architecture: str,
    seed: int,
    temp_stop: float,
    temp_anneal: bool,
    indicator_approx: Optional[str] = None,
    upperbound_method: Optional[str] = None,
    upperbound_param: Optional[int] = None,
    method: str = 'mc',
    comment: Optional[str] = None,
) -> str:
    """
    Generate a human-readable, unique run name.
    
    The name encodes key hyperparameters for easy identification:
    - Model type (exp/gs/exact)
    - Temperature settings
    - Annealing status
    - Seed
    
    Args:
        model_type: 'poisson' or 'gumbel_poisson'
        dataset: Dataset name (e.g., 'vH16')
        architecture: Architecture string (e.g., 'lin|lin')
        seed: Random seed
        temp_stop: Final temperature value
        temp_anneal: Whether temperature annealing is enabled
        indicator_approx: For Poisson, the indicator function type
        upperbound_method: For GumbelSoftmax, the upperbound method
        upperbound_param: For GumbelSoftmax, the upperbound parameter
        method: 'mc' or 'exact' for loss computation
        comment: Optional custom comment to append
        
    Returns:
        String name for the run (e.g., 'exp_sigmoid_t01_ann_seed0')
    """
    # Format temperature string
    temp_str = f"t{str(temp_stop).replace('.', '').replace('0', '', 1) if temp_stop < 1 else int(temp_stop)}"
    # Clean up common patterns
    if temp_str == 't1':
        temp_str = 't001'
    elif temp_stop == 0.01:
        temp_str = 't001'
    elif temp_stop == 0.1:
        temp_str = 't01'
    elif temp_stop == 0.3:
        temp_str = 't03'
    
    # Annealing string
    ann_str = 'ann' if temp_anneal else 'noann'
    
    # Model-specific prefix
    if model_type == 'poisson':
        if method == 'exact':
            prefix = f"exact_{indicator_approx or 'sigmoid'}"
        else:
            prefix = f"exp_{indicator_approx or 'sigmoid'}"
    elif model_type == 'gumbel_poisson':
        if upperbound_method == 'adaptive':
            prefix = 'gs_adaptive'
        else:
            prefix = f"gs_ub{upperbound_param or 50}"
    else:
        prefix = model_type
    
    # Combine parts
    parts = [prefix, temp_str, ann_str, f"seed{seed}"]
    
    # Add comment if provided
    if comment:
        parts.append(comment)
    
    return '_'.join(parts)


def create_checkpoint_dir(base_dir: str, run_name: str) -> Path:
    """
    Create checkpoint directory for a training run.
    
    Structure:
        checkpoints/sweep/{run_name}/
            - ConfigPoisVAE.json
            - ConfigTrainVAE.json
            - {Model}+TrainerVAE-{epoch}.pt
            - TrainerVAE.log
    
    Args:
        base_dir: Base checkpoint directory
        run_name: Name of the run
        
    Returns:
        Path to the created checkpoint directory
    """
    checkpoint_dir = Path(base_dir) / 'sweep' / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Train Poisson VAE models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --device 0 --model poisson --dataset vH16 --seed 0
  
  # With config file
  python train.py --config configs/poisson_default.yaml --device 0
  
  # Override config parameters
  python train.py --config configs/base.yaml --seed 42 --epochs 1000
  
  # Exact loss computation (linear decoder only)
  python train.py --device 0 --model poisson --method exact --dataset vH16
  
  # GumbelSoftmax variant
  python train.py --device 0 --model gumbel_poisson --upperbound_param 50
        """
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML/JSON configuration file')
    
    # Device
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device index or "cpu"')
    
    # Model architecture
    parser.add_argument('--model', type=str, default='poisson',
                        choices=['poisson', 'gumbel_poisson', 'gaussian', 'laplace', 'categorical'],
                        help='Model type to train')
    parser.add_argument('--dataset', type=str, default='vH16',
                        choices=['vH16', 'CIFAR16', 'MNIST', 'EMNIST', 'FashionMNIST', 'BALLS16'],
                        help='Dataset to use')
    parser.add_argument('--architecture', type=str, default='lin|lin',
                        help='Architecture: enc|dec (e.g., "lin|lin", "conv|lin")')
    parser.add_argument('--n_latents', type=int, default=512,
                        help='Number of latent dimensions')
    
    # Training method
    parser.add_argument('--method', type=str, default='mc',
                        choices=['mc', 'exact', 'score'],
                        help='Loss computation method (mc=Monte Carlo, exact=analytical, score=score matching)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Temperature annealing
    parser.add_argument('--temp_start', type=float, default=1.0,
                        help='Starting temperature')
    parser.add_argument('--temp_stop', type=float, default=0.1,
                        help='Final temperature')
    parser.add_argument('--temp_anneal', action='store_true',
                        help='Enable temperature annealing')
    parser.add_argument('--temp_anneal_portion', type=float, default=0.5,
                        help='Portion of training for temperature annealing')
    
    # KL annealing
    parser.add_argument('--kl_beta', type=float, default=1.0,
                        help='KL divergence weight (beta-VAE)')
    parser.add_argument('--kl_anneal_portion', type=float, default=0.5,
                        help='Portion of training for KL annealing')
    
    # Poisson-specific
    parser.add_argument('--indicator_approx', type=str, default='sigmoid',
                        choices=['sigmoid', 'cubic', 'linear', 'cosine'],
                        help='Soft indicator function for Poisson reparameterization')
    parser.add_argument('--prior_clamp', type=float, default=-4.0,
                        help='Prior log rate clamping value')
    
    # GumbelSoftmax-specific
    parser.add_argument('--upperbound_method', type=str, default='fixed',
                        choices=['fixed', 'std_ratio', 'quantile', 'adaptive'],
                        help='Method for computing GumbelSoftmax upperbound')
    parser.add_argument('--upperbound_param', type=int, default=50,
                        help='Parameter for upperbound method')
    
    # Output and logging
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Base directory for checkpoints')
    parser.add_argument('--comment', type=str, default=None,
                        help='Custom comment to append to run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='PoissonVAE',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default='poisson-collab',
                        help='W&B entity (team/user)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge config file settings with command-line arguments.
    
    CLI arguments override config file values.
    
    Args:
        config: Dictionary from config file
        args: Parsed command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Start with config file values
    merged = config.copy()
    
    # Get CLI args as dict, excluding None values and config path
    cli_dict = {k: v for k, v in vars(args).items() 
                if v is not None and k != 'config'}
    
    # CLI overrides config
    merged.update(cli_dict)
    
    return merged


def setup_training(config: Dict[str, Any]):
    """
    Set up and run training with the given configuration.
    
    This function:
    1. Creates model and trainer configurations
    2. Initializes the model
    3. Sets up the trainer with optional W&B logging
    4. Runs the training loop
    
    Args:
        config: Complete configuration dictionary
    """
    from main.config_vae import ConfigPoisVAE, ConfigTrainVAE
    from main.train_vae import TrainerVAE
    from main.vae import PoissonVAE, GumbelSoftmaxPoissonVAE, GaussianVAE, LaplaceVAE, CategoricalVAE
    
    # Parse architecture string
    enc_type, dec_type = config['architecture'].split('|')
    enc_type = enc_type.strip()
    dec_type = dec_type.strip()
    
    # Generate run name
    run_name = generate_run_name(
        model_type=config['model'],
        dataset=config['dataset'],
        architecture=config['architecture'],
        seed=config['seed'],
        temp_stop=config['temp_stop'],
        temp_anneal=config.get('temp_anneal', False),
        indicator_approx=config.get('indicator_approx'),
        upperbound_method=config.get('upperbound_method'),
        upperbound_param=config.get('upperbound_param'),
        method=config.get('method', 'mc'),
        comment=config.get('comment'),
    )
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(config['checkpoint_dir'], run_name)
    
    print("=" * 70)
    print(f"TRAINING: {run_name}")
    print("=" * 70)
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Method: {config.get('method', 'mc')}")
    print(f"Device: cuda:{config['device']}")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 70)
    
    # Build model config
    model_config_kwargs = dict(
        dataset=config['dataset'],
        n_latents=config.get('n_latents', 512),
        enc_type=enc_type,
        dec_type=dec_type,
        fit_prior=config.get('fit_prior', True),
        seed=config['seed'],  # seed is part of model config
        save=False,  # Don't auto-save, we'll save manually
    )
    
    # Add model-specific config
    if config['model'] == 'poisson':
        model_config_kwargs.update(
            indicator_approx=config.get('indicator_approx', 'sigmoid'),
            prior_clamp=config.get('prior_clamp', -4.0),
        )
    elif config['model'] == 'gumbel_poisson':
        model_config_kwargs.update(
            upperbound_method=config.get('upperbound_method', 'fixed'),
            upperbound_param=config.get('upperbound_param', 50),
        )
    
    # Create model config
    model_cfg = ConfigPoisVAE(**model_config_kwargs)
    
    # Create model
    model_classes = {
        'poisson': PoissonVAE,
        'gumbel_poisson': GumbelSoftmaxPoissonVAE,
        'gaussian': GaussianVAE,
        'laplace': LaplaceVAE,
        'categorical': CategoricalVAE,
    }
    ModelClass = model_classes[config['model']]
    model = ModelClass(model_cfg, verbose=config.get('verbose', False))
    
    # Build training config (seed is NOT in train config)
    train_config_kwargs = dict(
        epochs=config.get('epochs', 3000),
        batch_size=config.get('batch_size', 64),
        lr=config.get('lr', 2e-4),
        temp_start=config.get('temp_start', 1.0),
        temp_stop=config['temp_stop'],
        temp_anneal_portion=config.get('temp_anneal_portion', 0.5) if config.get('temp_anneal') else 0.0,
        kl_beta=config.get('kl_beta', 1.0),
        kl_anneal_portion=config.get('kl_anneal_portion', 0.5),
        method=config.get('method', 'mc'),
    )
    
    train_cfg = ConfigTrainVAE(**train_config_kwargs)
    
    # Add wandb settings (these are injected after config creation)
    train_cfg.no_wandb = config.get('no_wandb', False)
    train_cfg.wandb_project = config.get('wandb_project', 'PoissonVAE')
    train_cfg.wandb_entity = config.get('wandb_entity', 'poisson-collab')
    
    # Save configs to checkpoint directory
    _save_config_to_dir(model_cfg, checkpoint_dir, 'ConfigPoisVAE')
    _save_config_to_dir(train_cfg, checkpoint_dir, 'ConfigTrainVAE')
    
    # Setup device
    device = f"cuda:{config['device']}" if config['device'] != 'cpu' else 'cpu'
    
    # Create trainer
    trainer = TrainerVAE(
        model=model,
        cfg=train_cfg,
        device=device,
        verbose=config.get('verbose', False),
    )
    
    # Train
    trainer.train(fit_name=run_name)
    
    print("\n" + "=" * 70)
    print(f"✅ Training complete: {run_name}")
    print(f"   Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


def main():
    """Main entry point for training."""
    args = parse_arguments()
    
    # Load config file if provided
    if args.config:
        config = load_config_file(args.config)
        config = merge_config_with_args(config, args)
    else:
        # Use CLI args directly
        config = vars(args)
    
    # Run training
    setup_training(config)


if __name__ == '__main__':
    main()
