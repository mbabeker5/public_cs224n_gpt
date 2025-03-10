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

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
  ParaphraseDetectionDataset,
  ParaphraseDetectionTestDataset,
  load_paraphrase_data
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
    
    # Add attention pooling layer
    self.attention_pool = nn.Sequential(
        nn.Linear(self.hidden_size, 1),
        nn.Softmax(dim=1)
    )
    
    # Enhanced classification head with dropout and layer normalization
    self.classifier = nn.Sequential(
        nn.LayerNorm(self.hidden_size),
        nn.Dropout(0.1),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.GELU(),
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
    
    # Apply attention pooling over the sequence
    # First, mask out padding tokens
    extended_attention_mask = attention_mask.unsqueeze(-1)
    
    # Apply attention pooling to get a weighted sum of all token representations
    attention_weights = self.attention_pool(last_hidden_state)
    attention_weights = attention_weights * extended_attention_mask
    attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # Get weighted representation
    weighted_hidden_states = (last_hidden_state * attention_weights).sum(dim=1)
    
    # Also get the last token's hidden state for each sequence in the batch
    batch_size = input_ids.size(0)
    last_token_indices = attention_mask.sum(dim=1) - 1
    last_token_hidden_states = torch.stack([last_hidden_state[i, idx, :] 
                                           for i, idx in enumerate(last_token_indices)])
    
    # Combine the weighted representation with the last token representation
    combined_representation = weighted_hidden_states + last_token_hidden_states
    
    # Use our enhanced classifier to get better predictions
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


def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  # Create the data and its corresponding datasets and dataloader.
  para_train_data = load_paraphrase_data(args.para_train)
  para_dev_data = load_paraphrase_data(args.para_dev)

  para_train_data = ParaphraseDetectionDataset(para_train_data, args)
  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)

  para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=para_train_data.collate_fn)
  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)

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
  
  # Use a simpler learning rate scheduler for compatibility
  scheduler = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=0.1,
      end_factor=1.0,
      total_iters=warmup_steps
  )
  
  best_dev_acc = 0

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0
    optimizer.zero_grad()  # Zero gradients at the beginning of epoch
    for batch_idx, batch in enumerate(tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
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
      
      # Update weights only after accumulating enough gradients
      if (batch_idx + 1) % args.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(para_train_dataloader):
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate
        optimizer.zero_grad()  # Zero gradients after update

      train_loss += loss.item() * args.gradient_accumulation_steps  # Adjust for scaling
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

  model = ParaphraseGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()
  print(f"Loaded model to test from {args.filepath}")

  para_dev_data = load_paraphrase_data(args.para_dev)
  para_test_data = load_paraphrase_data(args.para_test, split='test')

  para_dev_data = ParaphraseDetectionDataset(para_dev_data, args)
  para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

  para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                   collate_fn=para_dev_data.collate_fn)
  para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
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

  parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=32)
  parser.add_argument("--lr", type=float, help="Learning rate", default=2e-5)
  parser.add_argument("--model_size", type=str,
                      help="The model size as specified on hugging face. DO NOT use the xl model.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large'], default='gpt2-medium')
  
  # Add parameter for gradient accumulation
  parser.add_argument("--gradient_accumulation_steps", type=int, help="Number of steps to accumulate gradients", default=2)
  
  # Add parameter to control how many layers to train
  parser.add_argument("--num_trainable_layers", type=int, 
                      help="Number of transformer layers to train (counting from the end). Set to 0 to train only the output layer, or -1 to train all layers.",
                      default=6)
  
  # Add parameter for weight decay to prevent overfitting
  parser.add_argument("--weight_decay", type=float, help="Weight decay for AdamW", default=0.01)
  
  # Add parameter for gradient clipping
  parser.add_argument("--max_grad_norm", type=float, help="Maximum gradient norm for clipping", default=1.0)

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
  args.filepath = f'{args.epochs}-{args.lr}-{args.num_trainable_layers}layers-para.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test(args)
