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
        # Heuristic strategy: Create challenging examples based on linguistic patterns
        for idx, (sent1, sent2, label, sent_id) in enumerate(train_data):
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(train_data)} samples")
                
            if label == 1:  # Paraphrase
                # For paraphrases, create a non-paraphrase that's similar but changes meaning
                # Use a mix of strategies for more diversity
                if random.random() < 0.33:
                    # Strategy 1: Modify key words to change meaning
                    modified_sent = modify_key_words(sent1)
                elif random.random() < 0.66:
                    # Strategy 2: Add negation or change quantifiers
                    modified_sent = add_negation_or_change_quantifiers(sent1)
                else:
                    # Strategy 3: Change the structure while preserving some words
                    modified_sent = restructure_with_preserved_words(sent1)
                
                losing_samples[sent_id] = (modified_sent, sent2, 0, f"{sent_id}_losing")
            else:  # Non-paraphrase
                # For non-paraphrases, create a paraphrase-like example that's actually not a paraphrase
                if random.random() < 0.5:
                    # Strategy 1: Make superficial changes that don't change meaning enough
                    modified_sent = make_superficial_changes(sent1)
                    losing_samples[sent_id] = (modified_sent, sent2, 1, f"{sent_id}_losing")
                else:
                    # Strategy 2: Create a sentence with similar words but different meaning
                    modified_sent = create_similar_but_different(sent1)
                    losing_samples[sent_id] = (modified_sent, sent2, 1, f"{sent_id}_losing")
    
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
                    incorrect_prob = probs[i, 1-label].item()
                    
                    # Create more challenging examples based on model confidence
                    if label == 1:  # Paraphrase
                        if correct_prob > 0.9:  # High confidence
                            # Create a very challenging example by preserving structure but changing meaning
                            modified_sent = preserve_structure_change_meaning(sent1)
                        else:  # Lower confidence
                            # Create a moderately challenging example
                            modified_sent = modify_key_words(sent1)
                        
                        losing_samples[sent_id] = (modified_sent, sent2, 0, f"{sent_id}_losing")
                    else:  # Non-paraphrase
                        if incorrect_prob > 0.3:  # Model is somewhat confused
                            # Create a very challenging example that looks like a paraphrase but isn't
                            modified_sent = create_deceptive_paraphrase(sent1)
                            losing_samples[sent_id] = (modified_sent, sent2, 1, f"{sent_id}_losing")
                        else:
                            # Create a moderately challenging example
                            modified_sent = make_superficial_changes(sent1)
                            losing_samples[sent_id] = (modified_sent, sent2, 1, f"{sent_id}_losing")
    
    print(f"Generated {len(losing_samples)} losing samples")
    return losing_samples

def modify_key_words(sentence):
    """Replace key words with antonyms or unrelated words to change meaning."""
    words = sentence.split()
    
    if len(words) < 3:
        return "not " + sentence
    
    # Identify potential key words (nouns, verbs, adjectives)
    # Simple heuristic: longer words are more likely to be content words
    content_word_indices = [i for i, word in enumerate(words) 
                           if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'have', 'what', 'when', 'where', 'which']]
    
    if not content_word_indices:
        return "not " + sentence
    
    # Replace 1-2 key words
    num_to_replace = min(len(content_word_indices), random.randint(1, 2))
    indices_to_replace = random.sample(content_word_indices, num_to_replace)
    
    # Word replacements that change meaning
    antonyms = {
        'good': 'bad', 'bad': 'good', 'high': 'low', 'low': 'high',
        'large': 'small', 'small': 'large', 'fast': 'slow', 'slow': 'fast',
        'hot': 'cold', 'cold': 'hot', 'new': 'old', 'old': 'new',
        'easy': 'difficult', 'difficult': 'easy', 'happy': 'sad', 'sad': 'happy',
        'right': 'wrong', 'wrong': 'right', 'true': 'false', 'false': 'true',
        'open': 'closed', 'closed': 'open', 'cheap': 'expensive', 'expensive': 'cheap',
        'increase': 'decrease', 'decrease': 'increase', 'buy': 'sell', 'sell': 'buy',
        'best': 'worst', 'worst': 'best', 'first': 'last', 'last': 'first',
        'love': 'hate', 'hate': 'love', 'start': 'finish', 'finish': 'start',
        'remember': 'forget', 'forget': 'remember', 'accept': 'reject', 'reject': 'accept'
    }
    
    # Generic replacements for words not in the antonym dictionary
    generic_replacements = ['different', 'opposite', 'unrelated', 'alternative', 
                           'wrong', 'incorrect', 'better', 'worse', 'never', 'always',
                           'rarely', 'frequently', 'unlikely', 'impossible', 'forbidden']
    
    for idx in indices_to_replace:
        word = words[idx].lower()
        if word in antonyms:
            words[idx] = antonyms[word]
        else:
            words[idx] = random.choice(generic_replacements)
    
    return ' '.join(words)

def add_negation_or_change_quantifiers(sentence):
    """Add negation or change quantifiers to alter meaning."""
    words = sentence.split()
    
    # Strategies:
    # 1. Add negation
    # 2. Change quantifiers (all -> some, some -> none, etc.)
    # 3. Add or remove modal verbs (can, must, should)
    
    strategy = random.choice(['negation', 'quantifier', 'modal'])
    
    if strategy == 'negation':
        # Add negation at the beginning or before a verb
        if random.random() < 0.5:
            return "not " + sentence
        else:
            # Find a verb-like position to insert negation
            for i in range(1, len(words)):
                if len(words[i]) > 3 and words[i].lower() not in ['this', 'that', 'with', 'from']:
                    words.insert(i, "not")
                    break
            else:
                # If no suitable position found, add at beginning
                words.insert(0, "not")
    
    elif strategy == 'quantifier':
        # Replace or add quantifiers
        quantifier_pairs = [
            ('all', 'some'), ('some', 'none'), ('every', 'few'), 
            ('many', 'few'), ('most', 'least'), ('always', 'never'),
            ('never', 'always'), ('everyone', 'no one'), ('everything', 'nothing')
        ]
        
        # Try to find and replace existing quantifiers
        replaced = False
        for i, word in enumerate(words):
            for q1, q2 in quantifier_pairs:
                if word.lower() == q1:
                    words[i] = q2
                    replaced = True
                    break
            if replaced:
                break
        
        # If no quantifier found, add one at the beginning
        if not replaced:
            words.insert(0, random.choice(['some', 'few', 'rarely', 'hardly']))
    
    elif strategy == 'modal':
        # Add or change modal verbs
        modal_pairs = [
            ('can', 'cannot'), ('must', 'need not'), ('should', 'should not'),
            ('will', 'will not'), ('may', 'may not'), ('could', 'could not')
        ]
        
        # Try to find and replace existing modals
        replaced = False
        for i, word in enumerate(words):
            for m1, m2 in modal_pairs:
                if word.lower() == m1:
                    words[i] = m2
                    replaced = True
                    break
            if replaced:
                break
        
        # If no modal found, add one at the beginning
        if not replaced:
            words.insert(0, random.choice(['cannot', 'should not', 'must not', 'will not']))
    
    return ' '.join(words)

def restructure_with_preserved_words(sentence):
    """Change the structure while preserving some key words."""
    words = sentence.split()
    
    if len(words) < 4:
        return "not " + sentence
    
    # Extract some key words (longer words are likely more important)
    key_words = [word for word in words if len(word) > 4]
    if not key_words:
        key_words = words
    
    # Select a subset of key words to preserve
    num_to_preserve = min(len(key_words), random.randint(2, 3))
    preserved_words = random.sample(key_words, num_to_preserve)
    
    # Create new structures with the preserved words
    structures = [
        f"Unlike {preserved_words[0]}, {' '.join(preserved_words[1:])} is completely different.",
        f"While {preserved_words[0]} might suggest {' '.join(preserved_words[1:])}, they are unrelated.",
        f"The concept of {preserved_words[0]} has nothing to do with {' '.join(preserved_words[1:])}.",
        f"Contrary to {preserved_words[0]}, {' '.join(preserved_words[1:])} means something else entirely.",
        f"Although {preserved_words[0]} sounds similar to {' '.join(preserved_words[1:])}, they differ in meaning."
    ]
    
    return random.choice(structures)

def make_superficial_changes(sentence):
    """Make superficial changes that don't significantly alter meaning."""
    words = sentence.split()
    
    if len(words) < 3:
        return sentence
    
    # Strategies:
    # 1. Reorder words slightly
    # 2. Add filler words
    # 3. Change word forms (singular/plural, tense)
    
    strategy = random.choice(['reorder', 'filler', 'form'])
    
    if strategy == 'reorder' and len(words) > 4:
        # Swap two adjacent words that aren't at the beginning
        idx = random.randint(1, len(words) - 2)
        words[idx], words[idx+1] = words[idx+1], words[idx]
    
    elif strategy == 'filler':
        # Add filler words that don't change meaning
        fillers = ['basically', 'essentially', 'fundamentally', 'generally', 
                  'actually', 'practically', 'virtually', 'technically']
        
        # Insert at a random position
        pos = random.randint(0, len(words))
        words.insert(pos, random.choice(fillers))
    
    elif strategy == 'form':
        # Try to change word forms
        # This is a simplified approach - in a real system you'd use NLP tools
        for i, word in enumerate(words):
            if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
                # Remove potential plural 's'
                words[i] = word[:-1]
                break
            elif len(word) > 3 and not word.endswith('s') and not word.endswith('sh') and not word.endswith('ch'):
                # Add potential plural 's'
                words[i] = word + 's'
                break
    
    return ' '.join(words)

def create_similar_but_different(sentence):
    """Create a sentence with similar words but different meaning."""
    words = sentence.split()
    
    if len(words) < 3:
        return "not " + sentence
    
    # Extract some key words
    key_words = [word for word in words if len(word) > 3]
    if not key_words:
        key_words = words
    
    # Select a subset of key words to use
    num_to_use = min(len(key_words), random.randint(2, 3))
    selected_words = random.sample(key_words, num_to_use)
    
    # Create new sentence with different meaning but similar words
    templates = [
        f"Is {selected_words[0]} related to {' '.join(selected_words[1:])}?",
        f"The difference between {selected_words[0]} and {' '.join(selected_words[1:])} is significant.",
        f"{selected_words[0]} cannot be compared with {' '.join(selected_words[1:])}.",
        f"While {selected_words[0]} exists, {' '.join(selected_words[1:])} is a separate concept.",
        f"How does {selected_words[0]} differ from {' '.join(selected_words[1:])}?"
    ]
    
    return random.choice(templates)

def preserve_structure_change_meaning(sentence):
    """Preserve sentence structure but change key content words to alter meaning."""
    words = sentence.split()
    
    if len(words) < 4:
        return "not " + sentence
    
    # Identify content words (nouns, verbs, adjectives)
    # Simple heuristic: longer words are more likely to be content words
    content_word_indices = [i for i, word in enumerate(words) 
                           if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'have', 'what', 'when', 'where', 'which']]
    
    if len(content_word_indices) < 2:
        return "not " + sentence
    
    # Replace most content words while keeping structure
    num_to_replace = max(2, len(content_word_indices) - 1)  # Replace all but one content word
    indices_to_replace = random.sample(content_word_indices, num_to_replace)
    
    # Domain-specific replacements to ensure meaning change
    replacements = [
        'computer', 'software', 'hardware', 'program', 'algorithm',
        'database', 'network', 'server', 'website', 'application',
        'interface', 'system', 'platform', 'framework', 'protocol',
        'language', 'code', 'function', 'variable', 'object',
        'method', 'class', 'library', 'module', 'package'
    ]
    
    for idx in indices_to_replace:
        words[idx] = random.choice(replacements)
    
    return ' '.join(words)

def create_deceptive_paraphrase(sentence):
    """Create a sentence that looks like a paraphrase but has a subtle meaning change."""
    words = sentence.split()
    
    if len(words) < 4:
        return sentence + " but not really"
    
    # Strategies:
    # 1. Change a single critical word that alters meaning
    # 2. Add a subtle qualifier that changes meaning
    # 3. Restructure to create a subtle meaning shift
    
    strategy = random.choice(['critical_word', 'qualifier', 'restructure'])
    
    if strategy == 'critical_word':
        # Find a critical word to change
        content_word_indices = [i for i, word in enumerate(words) 
                               if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from']]
        
        if content_word_indices:
            idx = random.choice(content_word_indices)
            # Replace with a word that looks similar but changes meaning
            similar_but_different = {
                'increase': 'decrease', 'decrease': 'increase',
                'include': 'exclude', 'exclude': 'include',
                'internal': 'external', 'external': 'internal',
                'maximum': 'minimum', 'minimum': 'maximum',
                'major': 'minor', 'minor': 'major',
                'positive': 'negative', 'negative': 'positive',
                'add': 'subtract', 'subtract': 'add',
                'before': 'after', 'after': 'before',
                'cause': 'effect', 'effect': 'cause',
                'create': 'destroy', 'destroy': 'create'
            }
            
            word = words[idx].lower()
            if word in similar_but_different:
                words[idx] = similar_but_different[word]
            else:
                # If no direct opposite, use a subtle replacement
                subtle_replacements = ['slightly', 'somewhat', 'partially', 'occasionally']
                words.insert(idx, random.choice(subtle_replacements))
    
    elif strategy == 'qualifier':
        # Add a subtle qualifier that changes meaning
        qualifiers = ['rarely', 'hardly', 'barely', 'seldom', 'occasionally', 
                     'supposedly', 'allegedly', 'seemingly', 'apparently']
        
        # Insert at a position that makes it look like a paraphrase
        pos = random.randint(0, min(3, len(words)))
        words.insert(pos, random.choice(qualifiers))
    
    elif strategy == 'restructure':
        # Restructure to create a subtle meaning shift
        if random.random() < 0.5:
            # Add a contradicting clause at the end
            contradiction = random.choice([
                "but not always", "though not in all cases", 
                "except in rare circumstances", "but with important differences",
                "although exceptions exist", "but only in theory"
            ])
            words.append(contradiction)
        else:
            # Change to a question form that subtly alters meaning
            question_prefix = random.choice([
                "Is it true that", "Could it be that", 
                "Do we know if", "Should we assume that"
            ])
            words = question_prefix.split() + words
    
    return ' '.join(words)
