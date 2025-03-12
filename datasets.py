# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv
import re
import torch
import random
import numpy as np

from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def preprocess_string(s):
  # Clean and normalize the text for better performance
  s = s.lower()
  s = re.sub(r'\s+', ' ', s)  # Replace multiple spaces with a single space
  s = re.sub(r'[^\w\s\?\.\,\']', ' ', s)  # Keep only alphanumeric, spaces, and some punctuation
  s = ' '.join(s.split())  # Normalize spaces
  s = s.replace('.', ' . ')
  s = s.replace('?', ' ? ')
  s = s.replace(',', ' , ')
  s = s.replace('\'', ' \' ')
  return s.strip()


class ParaphraseDetectionDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    # Convert labels to binary (0 or 1)
    labels = torch.LongTensor([x[2] for x in all_data])
    sent_ids = [x[3] for x in all_data]

    # Advanced prompt format with explicit instructions
    cloze_style_sents = [
        f'Task: Determine if two questions are paraphrases (same meaning, different words).\n'
        f'Question 1: "{s1}"\n'
        f'Question 2: "{s2}"\n'
        f'Analysis: Compare the core meaning of both questions.\n'
        f'Are these questions paraphrases of each other? '
        for (s1, s2) in zip(sent1, sent2)
    ]
    
    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True, 
                             max_length=256)  # Increased max length for better context

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'labels': labels,
      'sent_ids': sent_ids
    }

    return batched_data
    
  def collate_fn_dpo(self, all_data, losing_samples):
    """
    Collate function for DPO training that creates paired winning and losing samples.
    
    Args:
        all_data: Batch of original data
        losing_samples: Dictionary mapping from original sample ID to losing sample
        
    Returns:
        Dictionary containing winning and losing batches
    """
    # Process winning samples (original data)
    winning_batch = self.collate_fn(all_data)
    
    # Get the corresponding losing samples
    sent_ids = [x[3] for x in all_data]
    losing_data = []
    
    for sent_id in sent_ids:
        if sent_id in losing_samples:
            losing_data.append(losing_samples[sent_id])
        else:
            # If no losing sample exists, use a random one from the batch
            random_idx = random.randint(0, len(all_data) - 1)
            losing_data.append(all_data[random_idx])
    
    # Process losing samples
    losing_batch = self.collate_fn(losing_data)
    
    return {
        'winning': winning_batch,
        'losing': losing_batch
    }


class ParaphraseDetectionTestDataset(Dataset):
  def __init__(self, dataset, args):
    self.dataset = dataset
    self.p = args
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

  def collate_fn(self, all_data):
    sent1 = [x[0] for x in all_data]
    sent2 = [x[1] for x in all_data]
    sent_ids = [x[2] for x in all_data]

    # Advanced prompt format with explicit instructions
    cloze_style_sents = [
        f'Task: Determine if two questions are paraphrases (same meaning, different words).\n'
        f'Question 1: "{s1}"\n'
        f'Question 2: "{s2}"\n'
        f'Analysis: Compare the core meaning of both questions.\n'
        f'Are these questions paraphrases of each other? '
        for (s1, s2) in zip(sent1, sent2)
    ]

    encoding = self.tokenizer(cloze_style_sents, return_tensors='pt', padding=True, truncation=True,
                             max_length=256)  # Increased max length for better context

    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': sent_ids
    }

    return batched_data


def load_paraphrase_data(paraphrase_filename, split='train'):
  paraphrase_data = []
  if split == 'test':
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        sent_id = record['id'].lower().strip()
        # Apply preprocessing to improve data quality
        sent1 = preprocess_string(record['sentence1'])
        sent2 = preprocess_string(record['sentence2'])
        paraphrase_data.append((sent1, sent2, sent_id))

  else:
    with open(paraphrase_filename, 'r', encoding='utf-8') as fp:
      for record in csv.DictReader(fp, delimiter='\t'):
        try:
          sent_id = record['id'].lower().strip()
          # Apply preprocessing to improve data quality
          sent1 = preprocess_string(record['sentence1'])
          sent2 = preprocess_string(record['sentence2'])
          is_duplicate = int(float(record['is_duplicate']))
          
          # Add the original pair
          paraphrase_data.append((sent1, sent2, is_duplicate, sent_id))
          
          # Data augmentation: Add the reversed pair if this is the training set
          # This helps the model learn that order doesn't matter for paraphrasing
          if split == 'train':
            # Create a new ID for the reversed pair
            reversed_id = f"{sent_id}_rev"
            paraphrase_data.append((sent2, sent1, is_duplicate, reversed_id))
        except:
          pass

  print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")
  return paraphrase_data


class SonnetsDataset(Dataset):
  def __init__(self, file_path):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.sonnets = self._load_sonnets(file_path)

  def _load_sonnets(self, file_path):
    """Reads the file and extracts individual sonnets."""
    with open(file_path, 'r', encoding='utf-8') as f:
      text = f.read()

    # Split sonnets based on numbering pattern (e.g., "\n\n1\n\n")
    sonnets = re.split(r'\n\s*\d+\s*\n', text)[1:]  # Remove header text

    # Strip leading/trailing spaces
    return [s.strip() for s in sonnets]

  def __len__(self):
    return len(self.sonnets)

  def __getitem__(self, idx):
    return (idx, self.sonnets[idx])

  def collate_fn(self, all_data):
    idx = [example[0] for example in all_data]
    sonnets = [example[1] for example in all_data]

    encoding = self.tokenizer(sonnets, return_tensors='pt', padding=True, truncation=True)
    token_ids = torch.LongTensor(encoding['input_ids'])
    attention_mask = torch.LongTensor(encoding['attention_mask'])

    batched_data = {
      'token_ids': token_ids,
      'attention_mask': attention_mask,
      'sent_ids': idx
    }

    return batched_data

def generate_losing_samples(model, tokenizer, train_data, device, strategy="heuristic"):
    """
    Generate losing samples for DPO training.
    
    Args:
        model: The model to use for generating samples
        tokenizer: The tokenizer to use
        train_data: The training data
        device: The device to run the model on
        strategy: Strategy for generating losing samples ("heuristic" or "model_based")
        
    Returns:
        Dictionary mapping from original sample ID to losing sample
    """
    print(f"Generating losing samples using {strategy} strategy...")
    losing_samples = {}
    
    if strategy == "heuristic":
        # Heuristic strategy: Swap labels for non-paraphrases, modify sentences for paraphrases
        for idx, (sent1, sent2, label, sent_id) in enumerate(train_data):
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(train_data)} samples")
                
            if label == 1:  # Paraphrase
                # For paraphrases, modify one sentence to change meaning
                modified_sent = modify_sentence(sent1)
                losing_samples[sent_id] = (modified_sent, sent2, 0, f"{sent_id}_losing")
            else:  # Non-paraphrase
                # For non-paraphrases, just swap the label
                losing_samples[sent_id] = (sent1, sent2, 1, f"{sent_id}_losing")
    
    elif strategy == "model_based":
        # Model-based strategy: Use the model to find adversarial examples
        model.eval()
        
        # Create a dataset for batch processing
        temp_dataset = ParaphraseDetectionDataset(train_data, None)
        
        # Process in batches to avoid memory issues
        batch_size = 16
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(train_data))
                
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{num_batches}")
                
                # Get a batch of data
                batch_data = [train_data[i] for i in range(start_idx, end_idx)]
                batch = temp_dataset.collate_fn(batch_data)
                
                # Get model predictions
                token_ids = batch['token_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(token_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                # For each sample in the batch
                for i in range(len(batch_data)):
                    sent1, sent2, label, sent_id = batch_data[i]
                    
                    # Get the probability of the correct label
                    correct_prob = probs[i, label].item()
                    
                    if label == 1:  # Paraphrase
                        # For paraphrases, modify one sentence to change meaning
                        modified_sent = modify_sentence(sent1)
                        losing_samples[sent_id] = (modified_sent, sent2, 0, f"{sent_id}_losing")
                    else:  # Non-paraphrase
                        # For non-paraphrases with high confidence, swap the label
                        # For low confidence, find a more challenging example
                        if correct_prob > 0.8:
                            losing_samples[sent_id] = (sent1, sent2, 1, f"{sent_id}_losing")
                        else:
                            # Find a more challenging example by slightly modifying the sentences
                            modified_sent = modify_sentence(sent1, subtle=True)
                            losing_samples[sent_id] = (modified_sent, sent2, 1, f"{sent_id}_losing")
    
    print(f"Generated {len(losing_samples)} losing samples")
    return losing_samples

def modify_sentence(sentence, subtle=False):
    """
    Modify a sentence to change its meaning.
    
    Args:
        sentence: The sentence to modify
        subtle: Whether to make subtle changes (for more challenging examples)
        
    Returns:
        Modified sentence
    """
    words = sentence.split()
    
    if len(words) < 3:
        # For very short sentences, just add a negation
        return "not " + sentence
    
    if subtle:
        # Subtle modifications: change one word, add/remove a qualifier
        modifications = [
            # Change a random word
            lambda s: replace_random_word(s),
            # Add a qualifier
            lambda s: add_qualifier(s),
            # Remove a qualifier if present
            lambda s: remove_qualifier(s)
        ]
    else:
        # More significant modifications
        modifications = [
            # Add negation
            lambda s: "not " + s,
            # Change multiple words
            lambda s: replace_multiple_words(s),
            # Change the structure
            lambda s: restructure_sentence(s)
        ]
    
    # Choose a random modification
    modification = random.choice(modifications)
    return modification(sentence)

def replace_random_word(sentence):
    """Replace a random content word in the sentence."""
    words = sentence.split()
    if len(words) < 3:
        return sentence
    
    # Skip very short words and common stop words
    content_word_indices = [i for i, word in enumerate(words) 
                           if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from']]
    
    if not content_word_indices:
        return sentence
    
    idx_to_replace = random.choice(content_word_indices)
    
    # List of replacement words that could change meaning
    replacements = ['different', 'opposite', 'unrelated', 'similar', 'alternative', 
                   'wrong', 'correct', 'better', 'worse', 'never', 'always']
    
    words[idx_to_replace] = random.choice(replacements)
    return ' '.join(words)

def replace_multiple_words(sentence):
    """Replace multiple words in the sentence."""
    modified = replace_random_word(sentence)
    return replace_random_word(modified)

def add_qualifier(sentence):
    """Add a qualifier to the sentence."""
    qualifiers = ['sometimes', 'often', 'rarely', 'possibly', 'maybe', 
                 'occasionally', 'frequently', 'seldom', 'never', 'always']
    
    qualifier = random.choice(qualifiers)
    words = sentence.split()
    
    # Insert at beginning, middle, or end
    position = random.choice(['beginning', 'middle', 'end'])
    
    if position == 'beginning':
        return qualifier + ' ' + sentence
    elif position == 'middle' and len(words) > 2:
        mid_point = len(words) // 2
        words.insert(mid_point, qualifier)
        return ' '.join(words)
    else:
        return sentence + ' ' + qualifier

def remove_qualifier(sentence):
    """Remove a qualifier if present."""
    qualifiers = ['sometimes', 'often', 'rarely', 'possibly', 'maybe', 
                 'occasionally', 'frequently', 'seldom', 'never', 'always']
    
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in qualifiers]
    
    if len(filtered_words) < len(words):
        return ' '.join(filtered_words)
    else:
        # If no qualifier found, just return the original sentence
        return sentence

def restructure_sentence(sentence):
    """Change the structure of the sentence."""
    # Simple restructuring: add a prefix phrase
    prefixes = [
        "Contrary to popular belief, ",
        "Unlike what most people think, ",
        "In a different context, ",
        "From another perspective, ",
        "As an alternative view, "
    ]
    
    return random.choice(prefixes) + sentence
