'''
Run improved Direct Preference Optimization (DPO) training for paraphrase detection.

This script uses the modified paraphrase_detection.py with improved DPO implementation
to train a model that should achieve better performance than standard training.
It maintains compatibility with the original code and autograder.

Usage:
  python run_improved_dpo.py --use_gpu
'''

import os
import argparse
import subprocess
import torch
import random
import numpy as np
import sys

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_improved_dpo(args):
    """Run improved DPO training with the best settings."""
    # Create directories if they don't exist
    os.makedirs('dpo_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Build the command with optimized parameters
    # Use the exact same output paths as the original code for autograder compatibility
    cmd = [
        'python', 'paraphrase_detection.py',
        '--use_dpo',
        '--dpo_strategy', args.dpo_strategy,
        '--dpo_beta', str(args.dpo_beta),
        '--epochs', str(args.epochs),
        '--pretrain_epochs', str(args.pretrain_epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--num_trainable_layers', str(args.num_trainable_layers),
        '--weight_decay', str(args.weight_decay),
        '--max_grad_norm', str(args.max_grad_norm),
        '--seed', str(args.seed),
        '--para_dev_out', 'predictions/para-dev-output.csv',
        '--para_test_out', 'predictions/para-test-output.csv',
        '--model_size', args.model_size
    ]
    
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    if args.regenerate_losing_samples:
        cmd.append('--regenerate_losing_samples')
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Use subprocess.run with check=True to raise an exception if the command fails
        subprocess.run(cmd, check=True)
        
        # Check if the model was saved
        model_path = f'{args.epochs}-{args.lr}-{args.num_trainable_layers}layers-para.pt'
        if os.path.exists(model_path):
            print(f"Model saved successfully at {model_path}")
            
            # Check if prediction files were created
            if os.path.exists('predictions/para-dev-output.csv') and os.path.exists('predictions/para-test-output.csv'):
                print("Prediction files were created successfully.")
                print("The model should be compatible with the autograder.")
            else:
                print("Warning: Prediction files were not created. The model may not be compatible with the autograder.")
        else:
            print(f"Warning: Model file {model_path} not found. Check for errors in training.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=4, 
                       help="Number of epochs for DPO training")
    parser.add_argument("--pretrain_epochs", type=int, default=2,
                       help="Number of epochs to pretrain before DPO")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5, 
                       help="Learning rate")
    parser.add_argument("--use_gpu", action='store_true', 
                       help="Use GPU for training")
    parser.add_argument("--num_trainable_layers", type=int, default=8,
                       help="Number of transformer layers to train")
    parser.add_argument("--model_size", type=str, default="gpt2",
                       choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
                       help="The model size as specified on hugging face")
    
    # DPO-specific parameters
    parser.add_argument("--dpo_strategy", type=str, choices=["heuristic", "model_based"], 
                       default="heuristic", help="Strategy for generating losing samples")
    parser.add_argument("--dpo_beta", type=float, default=0.1, 
                       help="Temperature parameter for DPO loss")
    parser.add_argument("--regenerate_losing_samples", action='store_true',
                       help="Regenerate losing samples even if they exist in cache")
    
    # Other parameters
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for AdamW optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    
    print("Starting improved DPO training with the following settings:")
    print(f"  Model size: {args.model_size}")
    print(f"  Pretraining epochs: {args.pretrain_epochs}")
    print(f"  DPO training epochs: {args.epochs}")
    print(f"  DPO strategy: {args.dpo_strategy}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Using GPU: {args.use_gpu}")
    print(f"  Number of trainable layers: {args.num_trainable_layers}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max gradient norm: {args.max_grad_norm}")
    print(f"  Random seed: {args.seed}")
    print(f"  Regenerate losing samples: {args.regenerate_losing_samples}")
    
    run_improved_dpo(args)
    
    print("\nTraining complete. The model should be compatible with the autograder.")
    print("To verify, check that the following files exist:")
    print("  - predictions/para-dev-output.csv")
    print("  - predictions/para-test-output.csv")
    print(f"  - {args.epochs}-{args.lr}-{args.num_trainable_layers}layers-para.pt")

if __name__ == "__main__":
    main() 