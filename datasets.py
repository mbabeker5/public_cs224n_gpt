# !/usr/bin/env python3


"""
This file contains our Dataset class for Quora paraphrase detection. You may want to modify this file to train on
additional sources of data, or if you change how the Quora dataset is processed (i.e. data augmentation, etc.).
"""

import csv

import re
import torch

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
