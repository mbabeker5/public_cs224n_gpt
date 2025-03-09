'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    ### YOUR CODE HERE
    # Get the output from the GPT-2 model
    output = self.gpt(input_ids, attention_mask)
    
    # Extract the last hidden state - this contains hidden states for all tokens in the sequence
    hidden_states = output['last_hidden_state']
    
    # For language modeling, we need to predict the next token for each position
    # We use the GPT2's vocabulary size for the output dimension
    batch_size, seq_length, hidden_dim = hidden_states.shape
    
    # Project the hidden states to the vocabulary size using the GPT2's word embedding matrix
    # This is a common technique called weight tying
    logits = F.linear(hidden_states, self.gpt.word_embedding.weight)
    
    return logits


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  #Added repetition_penalty to discourage repetitive phrases and improve the quality of generated sonnets.
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128, repetition_penalty=1.2):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    # Track generated tokens for repetition penalty
    generated_tokens = token_ids[0].tolist()  # Start with initial tokens
    
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :].clone()
      
      # Apply repetition penalty - penalize tokens that appear in the last 50 tokens
      if repetition_penalty > 1.0:
        for token_id in set(generated_tokens[-50:]):  # Only consider recent tokens
          logits_last_token[:, token_id] /= repetition_penalty
      
      # Apply temperature scaling
      logits_last_token = logits_last_token / temperature

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      
      # Create mask for tokens within top_p probability mass
      sorted_indices_to_remove = cumulative_probs > top_p
      # Shift indices to keep first token above threshold
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0
      
      # Apply mask to probabilities
      indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
      probs = probs.masked_fill(indices_to_remove, 0.0)
      
      # Renormalize probabilities
      probs = probs / probs.sum(dim=-1, keepdim=True)

      # Sample from filtered distribution
      sampled_token = torch.multinomial(probs, 1)
      
      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Add sampled token to tracking list
      generated_tokens.append(sampled_token.item())
      
      # Append sampled token to input
      token_ids = torch.cat([token_ids, sampled_token.unsqueeze(0)], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")


def train(args):
  """Train GPT-2 for sonnet generation on the Shakespeare sonnets dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr)
  
  # Add learning rate scheduler
  total_steps = len(sonnet_dataloader) * args.epochs
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
  
  best_train_loss = float('inf')
  patience_counter = 0
  early_stop_patience = 5  # Increased patience from 3 to 5 to allow more training

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    
    # Check for early stopping
    if train_loss < best_train_loss:
      best_train_loss = train_loss
      patience_counter = 0
      # Save the best model
      save_model(model, optimizer, args, f'best_{args.filepath}')
    else:
      patience_counter += 1
      if patience_counter >= early_stop_patience:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break
    
    print('Generating several output sonnets...')
    model.eval()
    for batch in held_out_sonnet_dataset:
      encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  """
  Generate sonnets for submission by conditioning on the first 3 lines of each test sonnet.
  The model will complete the rest of the sonnet (lines 4-14).
  """
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Try to load the best model first, fall back to the last epoch model if not available
  try:
    saved = torch.load(f'best_{args.filepath}', weights_only=False)
    print(f"Using best model from best_{args.filepath}")
  except:
    saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
    print(f"Using last epoch model from {args.epochs-1}_{args.filepath}")

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    first_three_lines = batch[1]
    
    # Try multiple generations and pick the best one
    best_sonnet = None
    best_score = -float('inf')
    
    for _ in range(3):  # Generate 3 candidates
      # Tokenize the first three lines to condition the generation
      encoding = model.tokenizer(first_three_lines, return_tensors='pt', padding=False, truncation=True).to(device)
      
      # Generate the rest of the sonnet
      _, generated_text = model.generate(
          encoding['input_ids'], 
          temperature=args.temperature, 
          top_p=args.top_p,
          repetition_penalty=args.repetition_penalty
      )
      
      # Simple heuristic to score sonnets - prefer those with more line breaks and fewer repetitions
      line_count = generated_text.count('\n')
      repeated_phrases = sum(1 for phrase in generated_text.split() if generated_text.count(phrase) > 1 and len(phrase) > 3)
      score = line_count - (repeated_phrases * 0.5)
      
      if best_sonnet is None or score > best_score:
        best_sonnet = generated_text
        best_score = score
    
    # The generated_text includes the first three lines, so we keep the complete sonnet
    full_sonnet = best_sonnet
    
    generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'Sonnet {sonnet_id}:\n{full_sonnet}\n\n')

  # Write the generated sonnets to the output file
  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet_id, sonnet_text in generated_sonnets:
      f.write(f"\n{sonnet_id}\n")
      f.write(sonnet_text)
      f.write("\n\n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  # 15 epochs
  parser.add_argument("--epochs", type=int, default=15)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters - adjusted for better performance
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=0.7)  # Lower temperature 0.8 to 0.7  
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)  # Slightly lower top_p 0.9 to 0.92
  parser.add_argument("--repetition_penalty", type=float, help="Penalty to reduce repetition in generated text.",
                      default=1.3)  # Increased repetition penalty from 0.9 to 1.3

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  # 5e-5 learning rate
  parser.add_argument("--lr", type=float, help="learning rate", default=5e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  generate_submission_sonnets(args)