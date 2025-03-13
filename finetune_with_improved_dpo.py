'''
Fine-tune an existing model with improved Direct Preference Optimization (DPO).

This script loads a pre-trained model and fine-tunes it using the improved DPO implementation
while maintaining compatibility with the original code and autograder.

Usage:
  python finetune_with_improved_dpo.py --model_path <path_to_model> --use_gpu
'''

import os
import argparse
import subprocess
import torch
import random
import numpy as np
import sys
import shutil

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def finetune_with_improved_dpo(args):
    """Fine-tune an existing model with improved DPO."""
    # Create directories if they don't exist
    os.makedirs('dpo_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found.")
        sys.exit(1)
    
    # Load the model to extract its configuration
    try:
        model_info = torch.load(args.model_path, map_location='cpu')
        model_args = model_info['args']
        print(f"Loaded model from {args.model_path}")
        print(f"Original model parameters:")
        print(f"  Model size: {model_args.model_size}")
        print(f"  Learning rate: {model_args.lr}")
        print(f"  Number of trainable layers: {model_args.num_trainable_layers}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create a temporary copy of the model for safety
    temp_model_path = f"temp_{os.path.basename(args.model_path)}"
    shutil.copy2(args.model_path, temp_model_path)
    print(f"Created backup of original model at {temp_model_path}")
    
    # Build the command with optimized parameters
    # Use the exact same output paths as the original code for autograder compatibility
    cmd = [
        'python', 'paraphrase_detection.py',
        '--use_dpo',
        '--dpo_strategy', args.dpo_strategy,
        '--dpo_beta', str(args.dpo_beta),
        '--epochs', str(args.epochs),
        '--pretrain_epochs', '0',  # Skip pretraining since we're fine-tuning
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
        '--num_trainable_layers', str(model_args.num_trainable_layers),  # Use original model's value
        '--weight_decay', str(args.weight_decay),
        '--max_grad_norm', str(args.max_grad_norm),
        '--seed', str(args.seed),
        '--para_dev_out', 'predictions/para-dev-output.csv',
        '--para_test_out', 'predictions/para-test-output.csv',
        '--model_size', model_args.model_size  # Use original model's value
    ]
    
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    if args.regenerate_losing_samples:
        cmd.append('--regenerate_losing_samples')
    
    # Create a modified version of the model that can be loaded by paraphrase_detection.py
    # We need to rename it to match the expected filepath format
    expected_filepath = f'{args.epochs}-{args.lr}-{model_args.num_trainable_layers}layers-para.pt'
    
    # If a file with this name already exists, back it up
    if os.path.exists(expected_filepath):
        backup_path = f"backup_{expected_filepath}"
        shutil.move(expected_filepath, backup_path)
        print(f"Backed up existing model at {expected_filepath} to {backup_path}")
    
    # Copy the model to the expected filepath
    shutil.copy2(args.model_path, expected_filepath)
    print(f"Copied model to {expected_filepath} for fine-tuning")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Use subprocess.run with check=True to raise an exception if the command fails
        subprocess.run(cmd, check=True)
        
        # Check if the model was saved
        if os.path.exists(expected_filepath):
            # Get the modification time to see if it was updated
            orig_mtime = os.path.getmtime(temp_model_path)
            new_mtime = os.path.getmtime(expected_filepath)
            
            if new_mtime > orig_mtime:
                print(f"Model fine-tuned successfully at {expected_filepath}")
                
                # Check if prediction files were created
                if os.path.exists('predictions/para-dev-output.csv') and os.path.exists('predictions/para-test-output.csv'):
                    print("Prediction files were created successfully.")
                    print("The model should be compatible with the autograder.")
                else:
                    print("Warning: Prediction files were not created. The model may not be compatible with the autograder.")
            else:
                print(f"Warning: Model file {expected_filepath} was not updated. Fine-tuning may have failed.")
        else:
            print(f"Warning: Model file {expected_filepath} not found. Check for errors in fine-tuning.")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        
        # Restore the original model if fine-tuning failed
        if os.path.exists(temp_model_path):
            shutil.copy2(temp_model_path, args.model_path)
            print(f"Restored original model from backup.")
        
        sys.exit(1)
    
    # Clean up temporary files
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
        print(f"Removed temporary model file {temp_model_path}")
    
    # Create a copy with a more descriptive name
    descriptive_name = f"{os.path.splitext(args.model_path)[0]}_dpo_finetuned.pt"
    shutil.copy2(expected_filepath, descriptive_name)
    print(f"Created a copy of the fine-tuned model with a descriptive name: {descriptive_name}")

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model path
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the pre-trained model to fine-tune")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=2, 
                       help="Number of epochs for DPO fine-tuning")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-6, 
                       help="Learning rate (should be lower than original training)")
    parser.add_argument("--use_gpu", action='store_true', 
                       help="Use GPU for training")
    
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
    
    print("Starting improved DPO fine-tuning with the following settings:")
    print(f"  Model path: {args.model_path}")
    print(f"  DPO fine-tuning epochs: {args.epochs}")
    print(f"  DPO strategy: {args.dpo_strategy}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Using GPU: {args.use_gpu}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Max gradient norm: {args.max_grad_norm}")
    print(f"  Random seed: {args.seed}")
    print(f"  Regenerate losing samples: {args.regenerate_losing_samples}")
    
    finetune_with_improved_dpo(args)
    
    print("\nFine-tuning complete. The model should be compatible with the autograder.")
    print("To verify, check that the following files exist:")
    print("  - predictions/para-dev-output.csv")
    print("  - predictions/para-test-output.csv")

if __name__ == "__main__":
    main() 