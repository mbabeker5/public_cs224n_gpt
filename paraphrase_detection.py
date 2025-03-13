'''
Paraphrase detection for GPT starter code.

Consider:
 - ParaphraseGPT: Your implementation of the GPT-2 classification model.
 - train: Training procedure for ParaphraseGPT on the Quora paraphrase detection dataset.
 - test: Test procedure. This function generates the required files for your submission.

Running:
  `python paraphrase_detection.py --use_gpu`
trains and evaluates your ParaphraseGPT model and writes the required submission files.
'''

import argparse
import random
import torch
import gc
import os

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data,
  generate_losing_samples
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model
from transformers import GPT2Tokenizer

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


class ParaphraseGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    
    # By default, fine-tune the full model.
    # Instead of a 2-class output, we'll directly predict the token IDs for "yes" (8505) and "no" (3919)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Map class indices to token IDs
    self.class_token_ids = {
        0: 3919,  # "no" token ID
        1: 8505   # "yes" token ID
    }
    
    # Use args.d instead of config.n_embd since we're using a custom GPT2 implementation
    self.hidden_size = args.d
    
    # Advanced classification head with residual connections and layer normalization
    self.classifier = nn.Sequential(
        nn.LayerNorm(self.hidden_size),
        nn.Dropout(0.1),
        ResidualBlock(self.hidden_size),
        nn.LayerNorm(self.hidden_size),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_size, 2)
    )
    
    # We'll freeze most of the model by default to speed up training
    self.freeze_most_layers(args.num_trainable_layers)

  def freeze_most_layers(self, num_trainable_layers):
    """Freeze most of the model layers, only train the last few layers."""
    # First freeze all parameters
    for param in self.gpt.parameters():
      param.requires_grad = False
      
    # Then unfreeze only the last num_trainable_layers transformer blocks
    if num_trainable_layers > 0:
      # Get the number of layers from the config
      n_layers = self.gpt.config.num_hidden_layers
      
      # Unfreeze the last num_trainable_layers
      for i in range(max(0, n_layers - num_trainable_layers), n_layers):
        # The layers are in gpt_layers, not h
        for param in self.gpt.gpt_layers[i].parameters():
          param.requires_grad = True
      
      # Also unfreeze the output layer norm
      for param in self.gpt.final_layer_norm.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    TODO: Predict the label of the token using the paraphrase_detection_head Linear layer.

    We structure the input as:

      'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '

    So you want to find the prediction for the next token at the end of this sentence. Optimistically, it will be the
    token "yes" (byte pair encoding index of 8505) for examples that are paraphrases or "no" (byte pair encoding index
     of 3919) for examples that are not paraphrases.
    """

    'Takes a batch of sentences and produces embeddings for them.'
    ### YOUR CODE HERE
    # Get the output from the GPT-2 model
    output = self.gpt(input_ids, attention_mask)
    
    # Extract the last hidden state
    last_hidden_state = output['last_hidden_state']
    
    # Get the last token's hidden state for each sequence in the batch
    batch_size = input_ids.size(0)
    last_token_indices = attention_mask.sum(dim=1) - 1
    
    # Gather the hidden states for the last token in each sequence
    last_token_hidden_states = torch.stack([last_hidden_state[i, idx, :] 
                                           for i, idx in enumerate(last_token_indices)])
    
    # Advanced attention pooling
    # Create a learnable attention vector (implicitly through the linear layer)
    # that attends to different parts of the sequence
    attention_scores = torch.matmul(
        last_hidden_state, 
        last_token_hidden_states.unsqueeze(2)
    ).squeeze(2)
    
    # Apply mask to ignore padding tokens
    attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(2)
    
    # Apply attention weights to get context vector
    context_vector = torch.sum(last_hidden_state * attention_weights, dim=1)
    
    # Combine the last token representation with the context vector
    combined_representation = last_token_hidden_states + 0.5 * context_vector
    
    # Use our enhanced classifier to get predictions
    logits = self.classifier(combined_representation)
    
    return logits



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


def dpo_loss(model, winning_batch, losing_batch, beta=0.1, device="cuda", alpha=0.5):
  """
  Compute the DPO loss for a batch of winning and losing samples.
  
  Args:
      model: The model to train
      winning_batch: Batch of winning samples
      losing_batch: Batch of losing samples
      beta: Temperature parameter for the DPO loss
      device: The device to run the model on
      alpha: Weight for mixing DPO loss with standard CE loss
      
  Returns:
      DPO loss
  """
  # Get model outputs for winning samples
  winning_ids = winning_batch['token_ids'].to(device)
  winning_mask = winning_batch['attention_mask'].to(device)
  winning_logits = model(winning_ids, winning_mask)
  winning_labels = winning_batch['labels'].to(device)
  
  # Get model outputs for losing samples
  losing_ids = losing_batch['token_ids'].to(device)
  losing_mask = losing_batch['attention_mask'].to(device)
  losing_logits = model(losing_ids, losing_mask)
  losing_labels = losing_batch['labels'].to(device)
  
  # Compute standard cross-entropy loss for winning samples
  ce_loss = F.cross_entropy(winning_logits, winning_labels, reduction='mean')
  
  # Compute log probabilities for the correct labels
  winning_log_probs = F.log_softmax(winning_logits, dim=1)
  losing_log_probs = F.log_softmax(losing_logits, dim=1)
  
  # Extract log probs for the true labels
  winning_log_probs = torch.gather(winning_log_probs, 1, winning_labels.unsqueeze(1)).squeeze(1)
  losing_log_probs = torch.gather(losing_log_probs, 1, losing_labels.unsqueeze(1)).squeeze(1)
  
  # Compute the DPO loss with margin
  log_ratio = winning_log_probs - losing_log_probs
  dpo_component = -torch.mean(torch.log(torch.sigmoid(beta * log_ratio)))
  
  # Mix standard CE loss with DPO loss for more stable training
  # Start with more CE loss and gradually increase DPO component
  return alpha * ce_loss + (1 - alpha) * dpo_component


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  
  # Create output directory for DPO data if needed
  if args.use_dpo:
    os.makedirs('dpo_data', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
  
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  # First train a standard model if we're using DPO
  if args.use_dpo and args.pretrain_epochs > 0:
    print(f"Pre-training a standard model for {args.pretrain_epochs} epochs before DPO...")
    # Create standard dataset
    para_train_dataset = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_dataset = ParaphraseDetectionDataset(para_dev_data, args)
    
    para_train_dataloader = DataLoader(
      para_train_dataset, 
      shuffle=True, 
      batch_size=args.batch_size,
      collate_fn=para_train_dataset.collate_fn
    )
    
    para_dev_dataloader = DataLoader(
      para_dev_dataset, 
      shuffle=False, 
      batch_size=args.batch_size,
      collate_fn=para_dev_dataset.collate_fn
    )
    
    # Initialize model
    args_copy = add_arguments(args)
    model = ParaphraseGPT(args_copy)
    model = model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(para_train_dataloader) * args.pretrain_epochs // args.gradient_accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_dev_acc = 0
    
    # Pre-train for a few epochs
    for epoch in range(args.pretrain_epochs):
      model.train()
      train_loss = 0
      num_batches = 0
      optimizer.zero_grad()
      
      if torch.cuda.is_available():
          torch.cuda.empty_cache()
          gc.collect()
      
      for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'pretrain-{epoch}', disable=TQDM_DISABLE)):
        b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        labels = labels.to(device)

        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits, labels, reduction='mean')
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        loss_item = loss.item()
        
        if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(para_train_dataloader):
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
        
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if batch_idx % 500 == 0 and torch.cuda.is_available():
            del b_ids, b_mask, labels, logits, loss
            torch.cuda.empty_cache()

        train_loss += loss_item * args.gradient_accumulation_steps
        num_batches += 1

      train_loss = train_loss / num_batches
      dev_acc, dev_f1, _, _, _ = model_eval_paraphrase(para_dev_dataloader, model, device)

      if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        pretrain_path = f'pretrain-{args.pretrain_epochs}epochs-{args.lr}-para.pt'
        save_model(model, optimizer, args, pretrain_path)

      print(f"Pretrain Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")
    
    # Load the best pretrained model for DPO fine-tuning
    if best_dev_acc > 0:
      print(f"Loading best pretrained model with dev acc {best_dev_acc:.3f}")
      saved = torch.load(pretrain_path)
      model.load_state_dict(saved['model'])
    
    print("Starting DPO fine-tuning...")

  # If using DPO, generate losing samples
  if args.use_dpo:
    losing_samples_path = f'dpo_data/losing_samples_{args.dpo_strategy}.pt'
    if os.path.exists(losing_samples_path) and not args.regenerate_losing_samples:
      print(f"Loading losing samples from {losing_samples_path}")
      losing_samples = torch.load(losing_samples_path)
    else:
      print(f"Generating losing samples using strategy: {args.dpo_strategy}")
      tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
      
      # If we haven't pretrained, initialize a model for generating samples
      if not args.pretrain_epochs > 0 or not 'model' in locals():
        temp_args = add_arguments(args)
        temp_model = ParaphraseGPT(temp_args)
        temp_model = temp_model.to(device)
        model_for_samples = temp_model
      else:
        # Use the pretrained model for better sample generation
        model_for_samples = model
      
      losing_samples = generate_losing_samples(
          model_for_samples, 
          tokenizer, 
          para_train_data, 
          device, 
          strategy=args.dpo_strategy
      )
      torch.save(losing_samples, losing_samples_path)
      print(f"Saved losing samples to {losing_samples_path}")
      
      # Clean up temporary model if we created one
      if not args.pretrain_epochs > 0 or not 'model' in locals():
        del temp_model
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
          gc.collect()

  # Create datasets
  para_train_dataset = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_dataset = ParaphraseDetectionDataset(para_dev_data, args)

  # If using DPO, create paired dataset
  if args.use_dpo:
    para_train_dataloader = DataLoader(
      para_train_dataset, 
      shuffle=True, 
      batch_size=args.batch_size,
      collate_fn=lambda batch: para_train_dataset.collate_fn_dpo(batch, losing_samples)
    )
  else:
    para_train_dataloader = DataLoader(
      para_train_dataset, 
      shuffle=True, 
      batch_size=args.batch_size,
      collate_fn=para_train_dataset.collate_fn
    )
    
  para_dev_dataloader = DataLoader(para_dev_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=para_dev_dataset.collate_fn)

  # Initialize model if we haven't already from pretraining
  if not args.use_dpo or not args.pretrain_epochs > 0 or not 'model' in locals():
    args = add_arguments(args)
    model = ParaphraseGPT(args)
    model = model.to(device)

  # Print the number of trainable parameters
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Training {trainable_params:,} parameters out of {total_params:,} total parameters ({trainable_params/total_params:.2%})")

  lr = args.lr
  optimizer = AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
  
  # Add a learning rate scheduler for better convergence
  total_steps = len(para_train_dataloader) * args.epochs // args.gradient_accumulation_steps
  warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
  
  # Use a better learning rate scheduler for improved performance
  from transformers import get_linear_schedule_with_warmup
  scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=total_steps
  )
  
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()  # Zero gradients at the beginning of epoch
    
    # Clear CUDA cache at the beginning of each epoch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate alpha for DPO loss mixing (curriculum learning)
    # Start with more CE loss and gradually increase DPO component
    if args.use_dpo:
      alpha = max(0.1, 1.0 - (epoch / args.epochs))
    else:
      alpha = 0.5  # Not used for standard training
    
    for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
      # Different processing depending on whether we're using DPO
      if args.use_dpo:
        # Get winning and losing batches
        winning_batch = batch['winning']
        losing_batch = batch['losing']
        
        # Compute DPO loss with curriculum learning
        loss = dpo_loss(model, winning_batch, losing_batch, beta=args.dpo_beta, device=device, alpha=alpha)
      else:
        # Standard training
        # Get the input and move it to the gpu (I do not recommend training this model on CPU).
        b_ids, b_mask, labels = batch['token_ids'], batch['attention_mask'], batch['labels'].flatten()
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        labels = labels.to(device)

        # Compute the loss, gradients, and update the model's parameters.
        logits = model(b_ids, b_mask)
        
        # Apply cross entropy loss without label smoothing to ensure compatibility
        loss = F.cross_entropy(logits, labels, reduction='mean')
      
      # Scale the loss by gradient accumulation steps
      loss = loss / args.gradient_accumulation_steps
      loss.backward()
      
      # Store loss value before potentially deleting the tensor
      loss_item = loss.item()
      
      # Update weights only after accumulating enough gradients
      if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(para_train_dataloader):
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()  # Zero gradients after update
      
      # Free up memory periodically
      if batch_idx % 100 == 0:
        # Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
      
      # Delete tensors to free up memory only if we're close to OOM
      if batch_idx % 500 == 0 and torch.cuda.is_available():
          if args.use_dpo:
              del winning_batch, losing_batch, loss
          else:
              del b_ids, b_mask, labels, logits, loss
          torch.cuda.empty_cache()

      train_loss += loss_item * args.gradient_accumulation_steps  # Adjust for scaling
      num_batches += 1

    train_loss = train_loss / num_batches
    dev_acc, dev_f1, _, _, _ = model_eval_paraphrase(para_dev_dataloader, model, device)

    if dev_acc > best_dev_acc:
      best_dev_acc = dev_acc
      save_model(model, optimizer, args, args.filepath)

    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f}")


@torch.no_grad()
def test(args):
  """Evaluate your model on the dev and test datasets; save the predictions to disk."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(args.filepath)

  # Clear CUDA cache before loading model
  if torch.cuda.is_available():
      torch.cuda.empty_cache()
      gc.collect()

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  # Use a smaller batch size for evaluation to save memory
  eval_batch_size = min(args.batch_size, 4)
  
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=eval_batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=eval_batch_size,
                                    collate_fn=para_test_data.collate_fn)

  dev_para_acc, _, dev_para_y_pred, _, dev_para_sent_ids = model_eval_paraphrase(para_dev_dataloader, model, device)
  print(f"dev paraphrase acc :: {dev_para_acc :.3f}")
  test_para_y_pred, test_para_sent_ids = model_test_paraphrase(para_test_dataloader, model, device)

  with open(args.para_dev_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
      # Convert binary prediction to token ID (0 -> 3919, 1 -> 8505)
      token_id = 8505 if s == 1 else 3919
      f.write(f"{p}, {token_id} \n")

  with open(args.para_test_out, "w+") as f:
    f.write(f"id \t Predicted_Is_Paraphrase \n")
    for p, s in zip(test_para_sent_ids, test_para_y_pred):
      # Convert binary prediction to token ID (0 -> 3919, 1 -> 8505)
      token_id = 8505 if s == 1 else 3919
      f.write(f"{p}, {token_id} \n")


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
  parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
  parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
  parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
  parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--epochs", type=int, help="Number of epochs to train", default=6)
  parser.add_argument("--use_gpu", action='store_true')

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
  parser.add_argument("--lr", type=float, help="Learning rate", default=2e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2')
  
  # Add parameter for gradient accumulation
  parser.add_argument("--gradient_accumulation_steps", type=int, help="Number of steps to accumulate gradients", default=8)
  
  # Add parameter to control how many layers to train
  parser.add_argument("--num_trainable_layers", type=int, 
                      help="Number of transformer layers to train (counting from the end). Set to 0 to train only the output layer, or -1 to train all layers.",
                      default=8)
  
  # Add parameter for weight decay to prevent overfitting
  parser.add_argument("--weight_decay", type=float, help="Weight decay for AdamW", default=0.02)
  
  # Add parameter for gradient clipping
  parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm for clipping", default=1.0)
  
  # DPO-specific parameters
  parser.add_argument("--use_dpo", action='store_true', help="Use Direct Preference Optimization")
  parser.add_argument("--dpo_strategy", type=str, choices=["heuristic", "model_based"], default="heuristic",
                     help="Strategy for generating losing samples")
  parser.add_argument("--dpo_beta", type=float, default=0.1, 
                     help="Temperature parameter for DPO loss")
  parser.add_argument("--regenerate_losing_samples", action='store_true',
                     help="Regenerate losing samples even if they exist in cache")
  parser.add_argument("--pretrain_epochs", type=int, default=2,
                     help="Number of epochs to pretrain before DPO (0 to skip pretraining)")

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


# Residual block for better gradient flow
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual


if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-{args.num_trainable_layers}layers-para.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
