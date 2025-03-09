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
    
    # Add layer normalization for improved stability
    self.ln_f = nn.LayerNorm(args.d)
    
    # Initialize weights with a slightly different strategy
    self._init_weights()

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = True
      
  def _init_weights(self):
    """Initialize weights with a slightly different strategy for better performance"""
    # Initialize layer norm
    nn.init.normal_(self.ln_f.weight, mean=1.0, std=0.02)
    nn.init.constant_(self.ln_f.bias, 0.0)
    
    # Apply a small perturbation to the GPT2 word embedding for better initialization
    with torch.no_grad():
      word_embedding = self.gpt.word_embedding.weight
      # Add a small perturbation to break symmetry
      perturbation = torch.randn_like(word_embedding) * 0.02
      self.gpt.word_embedding.weight.copy_(word_embedding + perturbation)

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
    
    # Apply layer normalization for improved stability
    hidden_states = self.ln_f(hidden_states)
    
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
  def generate(self, encoding, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.

    UPDATE: Generates an original sonnet using top-p sampling, top-k sampling, and softmax temperature.
    Also implements repetition penalty to avoid repeating the same phrases.
    Includes length normalization and better stopping condition
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())

    # Track generated tokens for repetition penalty
    generated_tokens = token_ids[0].tolist()  # Start with initial tokens
    
    # Track line endings to enforce sonnet structure
    line_count = 0
    line_ending_tokens = [self.tokenizer.encode('\n')[0]]  # Newline token
    
    # Track rhyme pattern for sonnets (typically ABABCDCDEFEFGG for Shakespearean sonnets)
    # We'll use a simplified approach to encourage proper structure
    
    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Apply repetition penalty with decay based on distance
      if repetition_penalty > 1.0:
        for token_id in set(generated_tokens):
          # Check if token appears in recent history (more weight to very recent tokens)
          recent_occurrences = [i for i, t in enumerate(reversed(generated_tokens[-30:])) if t == token_id]
          if recent_occurrences:
            # Calculate penalty based on recency (more recent = stronger penalty)
            recency_weights = [1.0/(i+1) for i in recent_occurrences]
            recency_factor = sum(recency_weights)
            # Apply stronger penalty for frequently repeated tokens
            penalty = repetition_penalty * (1.0 + 0.3 * recency_factor)
            logits_last_token[0, token_id] /= penalty

      # Enhance probability of line endings after appropriate length
      # This helps maintain sonnet structure
      last_newline_pos = len(generated_tokens) - 1 - (generated_tokens[::-1].index(line_ending_tokens[0]) if line_ending_tokens[0] in generated_tokens else len(generated_tokens))
      tokens_since_newline = len(generated_tokens) - last_newline_pos - 1
      
      # Encourage line breaks at appropriate positions (roughly every 8-12 tokens)
      if tokens_since_newline > 10:
        for ending_token in line_ending_tokens:
          logits_last_token[0, ending_token] *= 1.5  # Boost probability of line endings
      
      # Top-k sampling (filter before top-p for efficiency)
      if top_k > 0:
        indices_to_remove = logits_last_token < torch.topk(logits_last_token, top_k)[0][..., -1, None]
        logits_last_token[indices_to_remove] = -float('Inf')

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding        
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Add the sampled token to our generated tokens list for repetition penalty
      generated_tokens.append(sampled_token.item())
      
      # Track line endings to maintain sonnet structure
      if sampled_token.item() in line_ending_tokens:
        line_count += 1
        # If we've generated 14 lines (a complete sonnet), consider stopping
        if line_count >= 14:
          # Add higher probability of stopping after 14 lines
          if random.random() < 0.8:
            break

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
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
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  
  # Add learning rate scheduler
  total_steps = len(sonnet_dataloader) * args.epochs
  warmup_steps = int(total_steps * args.warmup_ratio)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer, 
      T_0=warmup_steps, 
      T_mult=2, 
      eta_min=lr/10
  )
  
  # Add early stopping mechanism
  best_train_loss = float('inf')
  patience_counter = 0
  early_stop_patience = args.early_stop_patience
  
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
      
      # Apply gradient clipping
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, lr :: {scheduler.get_last_lr()[0]:.6f}")
    
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
      output = model.generate(
          encoding['input_ids'], 
          temperature=args.temperature, 
          top_p=args.top_p,
          top_k=args.top_k,
          repetition_penalty=args.repetition_penalty
      )
      print(f'{batch[1]}{output[1]}\n\n')

    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  """
  Generate sonnets for submission by conditioning on the first 3 lines of each test sonnet.
  The model will complete the rest of the sonnet (lines 4-14).
  Generates multiple candidates and selects the best one based on perplexity.
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
    
    # Generate multiple candidates
    candidates = []
    perplexities = []
    
    # Try different temperature and top_p combinations
    param_combinations = [
        {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_p": 0.92, "top_k": 40, "repetition_penalty": 1.3},
        {"temperature": 0.9, "top_p": 0.85, "top_k": 60, "repetition_penalty": 1.1},
        {"temperature": 1.0, "top_p": 0.88, "top_k": 45, "repetition_penalty": 1.25},
        {"temperature": 1.1, "top_p": 0.95, "top_k": 55, "repetition_penalty": 1.15}
    ]
    
    for params in param_combinations:
      encoding = model.tokenizer(first_three_lines, return_tensors='pt', padding=True, truncation=True).to(device)
      _, generated_text = model.generate(
          encoding['input_ids'], 
          temperature=params["temperature"], 
          top_p=params["top_p"],
          top_k=params["top_k"],
          repetition_penalty=params["repetition_penalty"]
      )
      
      # Calculate perplexity of the generated text
      full_sonnet = first_three_lines + generated_text
      tokens = model.tokenizer(full_sonnet, return_tensors='pt').to(device)
      with torch.no_grad():
          outputs = model(tokens['input_ids'], tokens['attention_mask'])
          shift_logits = outputs[..., :-1, :].contiguous()
          shift_labels = tokens['input_ids'][..., 1:].contiguous()
          loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
          perplexity = torch.exp(loss).item()
      
      candidates.append(generated_text)
      perplexities.append(perplexity)
    
    # Select the candidate with the lowest perplexity
    best_idx = perplexities.index(min(perplexities))
    best_candidate = candidates[best_idx]
    
    # Format the sonnet for submission
    generated_sonnets.append(f"{sonnet_id}\t{first_three_lines}{best_candidate}")

  # Write the generated sonnets to a file.
  with open(args.sonnet_out, 'w') as f:
    for sonnet in generated_sonnets:
      f.write(f"{sonnet}\n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.0)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)
  parser.add_argument("--top_k", type=int, help="Top-k sampling parameter", default=50)
  parser.add_argument("--repetition_penalty", type=float, help="Penalty for repeating tokens", default=1.2)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=8)
  parser.add_argument("--lr", type=float, help="learning rate", default=2e-5)
  parser.add_argument("--weight_decay", type=float, help="Weight decay for AdamW", default=0.01)
  parser.add_argument("--warmup_ratio", type=float, help="Ratio of total steps for warmup", default=0.1)
  parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm for clipping", default=1.0)
  parser.add_argument("--early_stop_patience", type=int, help="Patience for early stopping", default=3)
  
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
