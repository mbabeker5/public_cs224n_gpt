'''

Running:
  `python sonnet_dpo.py --use_gpu`

finetunes GPT2 with DPO

References: based on "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" and the reference implementation at https://github.com/eric-mitchell/direct-preference-optimization. 

Data generation: DeepSeakR1 model was used to generate the sonnet positive/negative pairs. based on the below criteria:
1- Poetic structure - Break the iambic pentameter structure that Shakespeare uses
2- Vocabulary richness - Replace rich, evocative words with more common, less colorful alternatives
3- Metaphorical complexity - Simplify or remove metaphors and similes
4- Rhyme scheme - Disrupt the rhyme scheme (Shakespeare's sonnets typically follow ABABCDCDEFEFGG)
5- Thematic coherence - Make some lines less connected to the sonnet's theme
6- Syntactic complexity - Simplify complex sentence structures
7- Emotional depth - Reduce the emotional nuance and make sentiments more direct/simplistic

'''

import argparse
import random
import torch
import numpy as np
import torch.nn.functional as F
import re
import os

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from sonnet_generation import SonnetGPT, add_arguments, seed_everything, save_model, generate_submission_sonnets

# Import the DPO loss function from the DPO repository
def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float) -> torch.FloatTensor:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.

    Returns:
        The DPO loss for each example in the batch.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    
    return losses


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    # Create a mask that is 1 for tokens we want to compute log probs for and 0 for others
    mask = (labels != -100)
    
    # Get the log probabilities for each token
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather the log probs at the positions of the labels
    # Make sure labels are within the valid range for gathering
    valid_labels = labels.clone()
    valid_labels[~mask] = 0  # Replace -100 with 0 to avoid out of bounds error
    
    log_probs_labels = torch.gather(log_probs, dim=-1, index=valid_labels.unsqueeze(-1)).squeeze(-1)
    
    # Apply the mask to only consider the tokens we care about
    log_probs_labels = log_probs_labels * mask
    
    # Sum the log probs for each sequence
    seq_log_probs = log_probs_labels.sum(dim=-1)
    
    if average_log_prob:
        # Compute the average log prob per token by dividing by the number of tokens
        token_count = mask.sum(dim=-1)
        # Avoid division by zero
        token_count = torch.maximum(token_count, torch.ones_like(token_count))
        seq_log_probs = seq_log_probs / token_count
    
    return seq_log_probs


class SonnetPairsDataset(Dataset):
    """Dataset for loading pairs of preferred and dispreferred sonnets."""
    
    def __init__(self, file_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.sonnet_pairs = self._load_sonnet_pairs(file_path)
        
    def _load_sonnet_pairs(self, file_path):
        """Reads the file and extracts pairs of original and degraded sonnets."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split the file by the separator
        sections = text.split("================================================================================")
        
        sonnet_pairs = []
        for section in sections:
            if not section.strip():
                continue
                
            # Extract original/preferred sonnet
            original_match = re.search(r'### ORIGINAL SONNET (\d+) ###(.*?)### DEGRADED SONNET', 
                                      section, re.DOTALL)
            if not original_match:
                continue
                
            sonnet_id = original_match.group(1)
            original_text = original_match.group(2).strip()
            
            # Extract degraded sonnet
            degraded_match = re.search(r'### DEGRADED SONNET \d+ ###(.*)', section, re.DOTALL)
            if not degraded_match:
                continue
                
            degraded_text = degraded_match.group(1).strip()
            
            sonnet_pairs.append({
                'id': sonnet_id,
                'chosen': original_text,  # Original sonnet is preferred
                'rejected': degraded_text  # Degraded sonnet is dispreferred
            })
        
        print(f"Loaded {len(sonnet_pairs)} sonnet pairs")
        return sonnet_pairs
    
    def __len__(self):
        return len(self.sonnet_pairs)
    
    def __getitem__(self, idx):
        return self.sonnet_pairs[idx]
    
    def collate_fn(self, batch):
        chosen_texts = [example['chosen'] for example in batch]
        rejected_texts = [example['rejected'] for example in batch]
        
        # Tokenize the chosen and rejected sonnets
        chosen_encodings = self.tokenizer(chosen_texts, return_tensors='pt', padding=True, truncation=True)
        rejected_encodings = self.tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True)
        
        # Create labels for computing log probabilities
        # Instead of shifting, we'll use the input_ids as labels but set the first token to -100 (ignore)
        chosen_labels = chosen_encodings['input_ids'].clone()
        chosen_labels[:, 0] = -100  # Ignore the first token (we don't predict it)
        
        rejected_labels = rejected_encodings['input_ids'].clone()
        rejected_labels[:, 0] = -100  # Ignore the first token (we don't predict it)
        
        return {
            'chosen_input_ids': chosen_encodings['input_ids'],
            'chosen_attention_mask': chosen_encodings['attention_mask'],
            'chosen_labels': chosen_labels,
            'rejected_input_ids': rejected_encodings['input_ids'],
            'rejected_attention_mask': rejected_encodings['attention_mask'],
            'rejected_labels': rejected_labels,
        }


def train_dpo(args):
    """Train GPT-2 for sonnet generation using DPO """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    # Create the data and its corresponding datasets and dataloader
    sonnet_pairs_dataset = SonnetPairsDataset(args.sonnet_pairs_path)
    sonnet_pairs_dataloader = DataLoader(
        sonnet_pairs_dataset, 
        shuffle=True, 
        batch_size=args.batch_size,
        collate_fn=sonnet_pairs_dataset.collate_fn
    )
    
    # Create the held-out dataset for generation
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    
    args = add_arguments(args)
    
    # Initialize the policy model (the one we're training)
    policy_model = SonnetGPT(args)
    policy_model = policy_model.to(device)
    
    # Initialize the reference model (frozen copy of the initial model)
    reference_model = SonnetGPT(args)
    reference_model = reference_model.to(device)
    
    # If a pre-trained model is provided, load it for both policy and reference models
    if args.pretrained_model:
        print(f"Loading pre-trained model from {args.pretrained_model}")
        try:
            saved = torch.load(args.pretrained_model, map_location=device)
            # Check if the model state is directly in 'model' key or nested in 'state'
            if 'model' in saved:
                policy_model.load_state_dict(saved['model'])
                reference_model.load_state_dict(saved['model'])
            elif 'state' in saved:
                policy_model.load_state_dict(saved['state'])
                reference_model.load_state_dict(saved['state'])
            else:
                policy_model.load_state_dict(saved)
                reference_model.load_state_dict(saved)
            print("xuccessfully loaded pre-trained model")
        except Exception as e:
            print(f"error loading model: {e}")
            print("continuing with randomly initialized model")
    
    # Freeze the reference model
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    
    # Set up optimizer for the policy model
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr)
    
    # Add early stopping mechanism
    best_train_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 3  # Stop after 3 epochs without improvement
    
    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        policy_model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(sonnet_pairs_dataloader, desc=f'train-dpo-{epoch}'):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Forward pass for chosen sonnets with policy model
            chosen_logits = policy_model(batch['chosen_input_ids'], batch['chosen_attention_mask'])
            policy_chosen_logps = _get_batch_logps(chosen_logits, batch['chosen_labels'])
            
            # Forward pass for rejected sonnets with policy model
            rejected_logits = policy_model(batch['rejected_input_ids'], batch['rejected_attention_mask'])
            policy_rejected_logps = _get_batch_logps(rejected_logits, batch['rejected_labels'])
            
            # Forward pass for chosen sonnets with reference model
            with torch.no_grad():
                ref_chosen_logits = reference_model(batch['chosen_input_ids'], batch['chosen_attention_mask'])
                reference_chosen_logps = _get_batch_logps(ref_chosen_logits, batch['chosen_labels'])
                
                # Forward pass for rejected sonnets with reference model
                ref_rejected_logits = reference_model(batch['rejected_input_ids'], batch['rejected_attention_mask'])
                reference_rejected_logps = _get_batch_logps(ref_rejected_logits, batch['rejected_labels'])
            
            # Compute DPO loss
            losses = preference_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=reference_chosen_logps,
                reference_rejected_logps=reference_rejected_logps,
                beta=args.beta
            )
            
            loss = losses.mean()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / num_batches
        print(f"Epoch {epoch}: DPO train loss :: {train_loss :.3f}.")
        
        # Check for early stopping
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            # Save the best model
            save_model(policy_model, optimizer, args, f'best_dpo_{args.filepath}')
            best_model = policy_model  # Keep track of the best model
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print('Generating several output sonnets...')
        policy_model.eval()
        for batch in held_out_sonnet_dataset:
            encoding = policy_model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
            output = policy_model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
            print(f'{batch[1]}{output[1]}\n\n')
        
        #  checkpoint
        save_model(policy_model, optimizer, args, f'{epoch}_dpo_{args.filepath}')
    
    # Return the best model 
    return best_model if 'best_model' in locals() else policy_model


def get_dpo_args():
    parser = argparse.ArgumentParser()
    
    # Data paths
    parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
    parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
    parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets_dpo.txt")
    parser.add_argument("--sonnet_pairs_path", type=str, default="data/sonnet_pairs_for_dpo.txt")
    
    # Training parameters
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--batch_size", help='The training batch size.', type=int, default=4)
    parser.add_argument("--lr", type=float, help="learning rate", default=5e-6)
    
    # DPO specific parameters
    parser.add_argument("--beta", type=float, help="Temperature parameter for DPO loss", default=0.1)
    parser.add_argument("--pretrained_model", type=str, help="Path to pretrained model", default=None)
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.0)
    parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
    
    # Model parameters
    parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_dpo_args()
    args.filepath = f'{args.epochs}-{args.lr}-{args.beta}-sonnet-dpo.pt'  # Save path
    seed_everything(args.seed)
    
    from datasets import SonnetsDataset
    
    best_model = train_dpo(args)
    
    # Generate sonnets with the DPO-trained model (directly using the best model)
    args.sonnet_out = "predictions/generated_sonnets_dpo.txt"  # Set output file
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    print("Generating sonnets with the best DPO model...")
    
    # Create the held-out dataset for generation
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)
    
    # Generate sonnets
    generated_sonnets = []
    for batch in held_out_sonnet_dataset:
        sonnet_id = batch[0]
        first_three_lines = batch[1]
        
        # Tokenize the first three lines to condition the generation
        encoding = best_model.tokenizer(first_three_lines, return_tensors='pt', padding=False, truncation=True).to(device)
        
        # Generate the rest of the sonnet
        _, generated_text = best_model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
        
        # The generated_text includes the first three lines, so we keep the complete sonnet
        full_sonnet = generated_text
        
        generated_sonnets.append((sonnet_id, full_sonnet))
        print(f'Sonnet {sonnet_id}:\n{full_sonnet}\n\n')
    
    # Write the generated sonnets to the output file
    with open(args.sonnet_out, "w+") as f:
        f.write(f"--Generated Sonnets with DPO-- \n\n")
        for sonnet_id, sonnet_text in generated_sonnets:
            f.write(f"\n{sonnet_id}\n")
            f.write(sonnet_text)
            f.write("\n\n") 