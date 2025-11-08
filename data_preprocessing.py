import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
import random
import numpy as np

class WikiTextDataset(Dataset):
    def __init__(self, 
                 split='train',
                 max_len=128,
                 mask_prob=0.15,
                 tokenizer=None):
        """
        Args:
            split: 'train' or 'validation'
            max_len: maximum sequence length
            mask_prob: probability of masking a token (default 15%)
            tokenizer: HuggingFace tokenizer
        """
        self.max_len = max_len
        self.mask_prob = mask_prob
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
            
        # Special tokens
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        
        print(f"Loading WikiText-2 {split} split...")
        # Load WikiText-2
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
        
        # Process into sentences
        self.sentences = self._process_text(dataset['text'])
        print(f"Processed {len(self.sentences)} sentences")
        
    def _process_text(self, texts):
        """Convert raw text into sentences"""
        sentences = []
        
        for text in texts:
            text = text.strip()
            # Skip empty lines and section headers (starting with =)
            if not text or text.startswith('='):
                continue
                
            # Split by periods, but keep sentences reasonable length
            splits = text.split('. ')
            for sent in splits:
                sent = sent.strip()
                if len(sent) > 10:  # Filter very short sentences
                    sentences.append(sent)
        
        return sentences
    
    def _create_masked_lm_predictions(self, tokens):
        """
        Create masked language model predictions following BERT strategy:
        - 80% of the time: replace with [MASK]
        - 10% of the time: replace with random word
        - 10% of the time: keep unchanged
        """
        output_tokens = tokens.copy()
        masked_positions = []
        masked_labels = []
        
        # Don't mask special tokens [CLS] and [SEP]
        # tokens[0] is [CLS], tokens[-1] or last non-pad is [SEP]
        maskable_positions = list(range(1, len(tokens) - 1))
        
        # Determine which positions to mask
        num_to_mask = max(1, int(len(maskable_positions) * self.mask_prob))
        mask_positions = random.sample(maskable_positions, 
                                      min(num_to_mask, len(maskable_positions)))
        
        for pos in mask_positions:
            masked_positions.append(pos)
            masked_labels.append(tokens[pos])
            
            prob = random.random()
            if prob < 0.8:
                # 80% - replace with [MASK]
                output_tokens[pos] = self.mask_token_id
            elif prob < 0.9:
                # 10% - replace with random token
                output_tokens[pos] = random.randint(0, self.vocab_size - 1)
            # else: 10% - keep original (do nothing)
        
        return output_tokens, masked_positions, masked_labels
    
    def _create_sentence_pair(self, idx):
        """
        Create a sentence pair for NSP task
        50% of the time: use actual next sentence (is_next=1)
        50% of the time: use random sentence (is_next=0)
        """
        sentence_a = self.sentences[idx]
        
        # 50% chance of using next sentence
        if random.random() < 0.5 and idx < len(self.sentences) - 1:
            # Use next sentence
            sentence_b = self.sentences[idx + 1]
            is_next = 1
        else:
            # Use random sentence
            random_idx = random.randint(0, len(self.sentences) - 1)
            while random_idx == idx or random_idx == idx + 1:
                random_idx = random.randint(0, len(self.sentences) - 1)
            sentence_b = self.sentences[random_idx]
            is_next = 0
        
        return sentence_a, sentence_b, is_next
    
    def __len__(self):
        return len(self.sentences) - 1  # -1 to ensure we can get next sentence
    
    def __getitem__(self, idx):
        """
        Returns a training example with:
        - input_ids: tokenized sequence with [CLS] and [SEP]
        - segment_ids: 0 for sentence A, 1 for sentence B
        - masked_positions: positions of masked tokens
        - masked_labels: original tokens at masked positions
        - is_next: NSP label
        - attention_mask: mask for padding
        """
        # Get sentence pair
        sent_a, sent_b, is_next = self._create_sentence_pair(idx)
        
        # Tokenize
        tokens_a = self.tokenizer.tokenize(sent_a)
        tokens_b = self.tokenizer.tokenize(sent_b)
        
        # Truncate if too long (reserve space for [CLS], [SEP], [SEP])
        max_tokens_per_sent = (self.max_len - 3) // 2
        if len(tokens_a) > max_tokens_per_sent:
            tokens_a = tokens_a[:max_tokens_per_sent]
        if len(tokens_b) > max_tokens_per_sent:
            tokens_b = tokens_b[:max_tokens_per_sent]
        
        # Combine: [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        
        # Convert to ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create masked LM predictions
        masked_input_ids, masked_positions, masked_labels = \
            self._create_masked_lm_predictions(input_ids)
        
        # Padding
        seq_len = len(masked_input_ids)
        attention_mask = [1] * seq_len
        
        padding_len = self.max_len - seq_len
        masked_input_ids += [self.pad_token_id] * padding_len
        segment_ids += [0] * padding_len
        attention_mask += [0] * padding_len
        
        # Pad masked positions and labels to fixed size
        max_masked = 20  # Maximum number of masked positions
        num_masked = len(masked_positions)
        masked_positions += [0] * (max_masked - num_masked)
        masked_labels += [-1] * (max_masked - num_masked)  # -1 will be ignored in loss
        
        # Convert to tensors
        return {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'masked_positions': torch.tensor(masked_positions[:max_masked], dtype=torch.long),
            'masked_labels': torch.tensor(masked_labels[:max_masked], dtype=torch.long),
            'is_next': torch.tensor(is_next, dtype=torch.long)
        }


def create_dataloaders(batch_size=16, max_len=128, num_workers=2):
    """
    Create train and validation dataloaders
    """
    print("Creating datasets...")
    train_dataset = WikiTextDataset(split='train', max_len=max_len)
    val_dataset = WikiTextDataset(split='validation', max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    return train_loader, val_loader, train_dataset.tokenizer


# Testing and demonstration
if __name__ == "__main__":
    print("="*60)
    print("Testing WikiText-2 Data Preprocessing")
    print("="*60)
    
    # Create dataset
    dataset = WikiTextDataset(split='train', max_len=128)
    
    # Get a sample
    sample = dataset[0]
    
    print("\n--- Sample Data ---")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Segment IDs shape: {sample['segment_ids'].shape}")
    print(f"Attention Mask shape: {sample['attention_mask'].shape}")
    print(f"Masked Positions shape: {sample['masked_positions'].shape}")
    print(f"Masked Labels shape: {sample['masked_labels'].shape}")
    print(f"Is Next: {sample['is_next'].item()}")
    
    # Decode to see actual tokens
    tokenizer = dataset.tokenizer
    input_tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist())
    
    # Find non-padding tokens
    non_pad = sample['attention_mask'].sum().item()
    
    print(f"\n--- Decoded Tokens (first {non_pad} tokens) ---")
    print(' '.join(input_tokens[:non_pad]))
    
    # Show masked positions
    masked_pos = sample['masked_positions']
    masked_labels = sample['masked_labels']
    
    print("\n--- Masked Tokens ---")
    for i, (pos, label) in enumerate(zip(masked_pos, masked_labels)):
        if label.item() == -1:  # Padding
            break
        pos_val = pos.item()
        label_val = label.item()
        current_token = input_tokens[pos_val]
        original_token = tokenizer.convert_ids_to_tokens([label_val])[0]
        print(f"Position {pos_val}: '{current_token}' (original: '{original_token}')")
    
    # Test dataloader
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    train_loader, val_loader, tokenizer = create_dataloaders(batch_size=4, max_len=128)
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print("\n--- Batch Shapes ---")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")
    
    print("\n--- Batch Statistics ---")
    print(f"NSP labels in batch: {batch['is_next'].tolist()}")
    print(f"Average sequence length: {batch['attention_mask'].sum(dim=1).float().mean():.1f}")
    
    # Count actual masked tokens (non -1)
    num_masked = (batch['masked_labels'] != -1).sum(dim=1)
    print(f"Masked tokens per example: {num_masked.tolist()}")
    
    print("\n✓ Data preprocessing is working correctly!")
    print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    print(f"✓ [CLS] token ID: {tokenizer.cls_token_id}")
    print(f"✓ [SEP] token ID: {tokenizer.sep_token_id}")
    print(f"✓ [MASK] token ID: {tokenizer.mask_token_id}")
    print(f"✓ [PAD] token ID: {tokenizer.pad_token_id}")