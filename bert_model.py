import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_segments,
                 max_len,
                 embed_dim,
                 dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.seg_embed = nn.Embedding(n_segments, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, seq, seg):
        # seq: (batch_size, seq_len)
        # seg: (batch_size, seq_len)
        batch_size, seq_len = seq.size()
        
        # Create position indices on the same device as input
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        
        # Sum embeddings
        embed_val = self.tok_embed(seq) + self.seg_embed(seg) + self.pos_embed(pos)
        embed_val = self.norm(embed_val)
        embed_val = self.drop(embed_val)
        return embed_val


class BERT(nn.Module):
    def __init__(self,
                 vocab_size,
                 n_segments,
                 max_len,
                 embed_dim,
                 n_layers,
                 attn_heads,
                 dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding layer
        self.embedding = BERTEmbedding(vocab_size, n_segments, max_len, embed_dim, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=attn_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # Important: batch_first=True
        )
        self.encoder_block = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # MLM Head (Masked Language Model)
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )
        
        # NSP Head (Next Sentence Prediction)
        self.nsp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2)  # Binary classification
        )
        
    def forward(self, seq, seg, mask=None):
        """
        Args:
            seq: (batch_size, seq_len) - token ids
            seg: (batch_size, seq_len) - segment ids (0 or 1)
            mask: (batch_size, seq_len) - padding mask (optional)
        Returns:
            mlm_output: (batch_size, seq_len, vocab_size)
            nsp_output: (batch_size, 2)
        """
        # Get embeddings
        embedded = self.embedding(seq, seg)
        
        # Pass through encoder
        # Create attention mask if provided (True = ignore, False = attend)
        if mask is not None:
            # TransformerEncoder expects True for positions to ignore
            src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
            
        encoded = self.encoder_block(embedded, src_key_padding_mask=src_key_padding_mask)
        
        # MLM: predict all tokens
        mlm_output = self.mlm_head(encoded)
        
        # NSP: use [CLS] token (first token)
        cls_output = encoded[:, 0, :]  # (batch_size, embed_dim)
        nsp_output = self.nsp_head(cls_output)
        
        return mlm_output, nsp_output


class BERTForPretraining(nn.Module):
    """Wrapper with loss computation"""
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        
    def forward(self, seq, seg, masked_pos, masked_tokens, is_next, mask=None):
        """
        Args:
            seq: (batch_size, seq_len) - input token ids
            seg: (batch_size, seq_len) - segment ids
            masked_pos: (batch_size, num_masked) - positions of masked tokens
            masked_tokens: (batch_size, num_masked) - true tokens at masked positions
            is_next: (batch_size,) - NSP labels (1 if next sentence, 0 if random)
            mask: (batch_size, seq_len) - padding mask
        Returns:
            total_loss, mlm_loss, nsp_loss
        """
        mlm_output, nsp_output = self.bert(seq, seg, mask)
        
        batch_size, seq_len, vocab_size = mlm_output.size()
        
        # MLM Loss - only compute loss on masked positions
        # Gather predictions at masked positions
        mlm_output_flat = mlm_output.view(-1, vocab_size)
        
        # Create indices for gathering
        batch_indices = torch.arange(batch_size, device=seq.device).unsqueeze(1).expand_as(masked_pos)
        flat_masked_pos = (batch_indices * seq_len + masked_pos).view(-1)
        
        # Gather predictions at masked positions
        masked_predictions = mlm_output_flat[flat_masked_pos]
        masked_tokens_flat = masked_tokens.view(-1)
        
        # Compute MLM loss
        mlm_loss = F.cross_entropy(masked_predictions, masked_tokens_flat, ignore_index=-1)
        
        # NSP Loss
        nsp_loss = F.cross_entropy(nsp_output, is_next)
        
        # Total loss
        total_loss = mlm_loss + nsp_loss
        
        return total_loss, mlm_loss, nsp_loss


# Testing code
if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 30000
    N_SEGMENTS = 2  # Changed to 2 (sentence A=0, sentence B=1)
    MAX_LEN = 256  # Reduced for faster training
    EMBED_DIM = 512  # Reduced from 768
    N_LAYERS = 6  # Reduced from 12
    ATTN_HEADS = 8  # Reduced from 12
    DROPOUT = 0.1
    BATCH_SIZE = 8
    
    # Special token IDs (define these based on your tokenizer)
    CLS_TOKEN_ID = 101
    SEP_TOKEN_ID = 102
    MASK_TOKEN_ID = 103
    PAD_TOKEN_ID = 0
    
    # Create model
    bert = BERT(VOCAB_SIZE, N_SEGMENTS, MAX_LEN, EMBED_DIM, N_LAYERS, ATTN_HEADS, DROPOUT)
    bert_pretraining = BERTForPretraining(bert)
    
    # Create dummy batch
    sample_seq = torch.randint(high=VOCAB_SIZE, size=[BATCH_SIZE, MAX_LEN])
    sample_seg = torch.randint(high=N_SEGMENTS, size=[BATCH_SIZE, MAX_LEN])
    
    # Create masked positions (15% masking, ~19 tokens per sequence)
    num_masked = 19
    masked_pos = torch.randint(1, MAX_LEN-1, size=[BATCH_SIZE, num_masked])
    masked_tokens = torch.randint(high=VOCAB_SIZE, size=[BATCH_SIZE, num_masked])
    
    # NSP labels
    is_next = torch.randint(0, 2, size=[BATCH_SIZE])
    
    # Padding mask (1 for real tokens, 0 for padding)
    mask = torch.ones(BATCH_SIZE, MAX_LEN)
    
    # Test forward pass
    print("Testing BERT model...")
    mlm_output, nsp_output = bert(sample_seq, sample_seg, mask)
    print(f"MLM output shape: {mlm_output.shape}")  # (batch_size, seq_len, vocab_size)
    print(f"NSP output shape: {nsp_output.shape}")  # (batch_size, 2)
    
    # Test with loss computation
    print("\nTesting with loss computation...")
    total_loss, mlm_loss, nsp_loss = bert_pretraining(
        sample_seq, sample_seg, masked_pos, masked_tokens, is_next, mask
    )
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"MLM Loss: {mlm_loss.item():.4f}")
    print(f"NSP Loss: {nsp_loss.item():.4f}")
    
    print("\n✓ Model architecture is working correctly!")
    print(f"\nModel Parameters: {sum(p.numel() for p in bert.parameters()):,}")