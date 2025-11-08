# BERT Implementation from Scratch

A PyTorch implementation of BERT (Bidirectional Encoder Representations from Transformers) trained on WikiText-2 corpus using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) objectives.

## Project Overview

This project implements a simplified version of BERT following the original paper ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)). The model is trained from scratch on the WikiText-2 dataset to learn contextual representations of language through two pre-training tasks:

1. **Masked Language Modeling (MLM)**: Predicting randomly masked tokens in a sequence
2. **Next Sentence Prediction (NSP)**: Determining if two sentences are consecutive or random

## Architecture

### Model Configuration

The implementation uses a scaled-down version of BERT to accommodate training constraints:

- **Vocabulary Size**: 30,522 (BERT base uncased tokenizer)
- **Embedding Dimension**: 384
- **Number of Layers**: 6 transformer encoder layers
- **Attention Heads**: 8 per layer
- **Feed-Forward Dimension**: 1024 (4x embedding dimension)
- **Maximum Sequence Length**: 128 tokens
- **Dropout**: 0.1
- **Total Parameters**: ~34M parameters

### Key Components

#### 1. BERT Embedding Layer (`BERTEmbedding`)
Combines three types of embeddings:
- **Token Embeddings**: Learned representations for each token in vocabulary
- **Position Embeddings**: Encodes token position in sequence
- **Segment Embeddings**: Distinguishes between sentence A (0) and sentence B (1)

All embeddings are summed, normalized, and passed through dropout.

#### 2. Transformer Encoder (`BERT`)
- Multi-layer transformer encoder with self-attention mechanism
- Uses PyTorch's `nn.TransformerEncoder` with GELU activation
- Processes sequences bidirectionally to capture context from both directions

#### 3. Output Heads
- **MLM Head**: Projects encoded representations to vocabulary size for token prediction
- **NSP Head**: Binary classifier using [CLS] token representation for sentence pair classification

## Dataset

### WikiText-2
- **Source**: HuggingFace datasets (`wikitext-2-v1`)
- **Training Set**: ~2M tokens, ~36K sentences
- **Validation Set**: ~217K tokens, ~3.7K sentences
- **Preprocessing**:
  - Sentences extracted from Wikipedia articles
  - Empty lines and section headers filtered out
  - Minimum sentence length: 10 characters

## Training Procedure

### Masked Language Modeling (MLM)
Following BERT's masking strategy:
- **15%** of tokens randomly selected for masking
- Of the selected tokens:
  - **80%**: Replaced with `[MASK]` token
  - **10%**: Replaced with random token
  - **10%**: Kept unchanged

This prevents the model from only learning the `[MASK]` token and encourages robust representations.

### Next Sentence Prediction (NSP)
- **Positive examples (50%)**: Sentence B actually follows sentence A in the corpus
- **Negative examples (50%)**: Sentence B is randomly sampled from elsewhere

Input format: `[CLS] Sentence A [SEP] Sentence B [SEP]`

### Training Configuration
- **Optimizer**: AdamW (learning rate: 3e-5, weight decay: 0.01)
- **Scheduler**: Linear warmup + decay (1000 warmup steps)
- **Batch Size**: 12
- **Epochs**: 30
- **Gradient Clipping**: Max norm 1.0
- **Loss**: Combined MLM loss + NSP loss

## Usage

### 1. Training the Model

```bash
python bert_training.py
```

This will:
- Load and preprocess WikiText-2 dataset
- Initialize BERT model
- Train for 30 epochs with validation after each epoch
- Save checkpoints in `checkpoints/` directory
- Generate training curves plot
- Display example predictions

**Outputs:**
- `checkpoints/best_model.pt`: Best model based on validation loss
- `checkpoints/final_model.pt`: Final model after all epochs
- `checkpoints/training_history.json`: Loss and accuracy history
- `checkpoints/training_curves.png`: Training/validation curves

### 2. Evaluation and Demonstration

```bash
python bert_eval.py
```

Interactive menu options:
1. **Masked Language Modeling Demo**: See top predictions for masked tokens
2. **Next Sentence Prediction Demo**: Test sentence pair classification
3. **Full Validation Evaluation**: Compute metrics on validation set
4. **All of the above**: Run complete evaluation
5. **Custom sentence MLM**: Input your own sentences with `[MASK]` tokens

## File Structure

```
.
├── bert_model.py              # Model architecture
│   ├── BERTEmbedding         # Token + Position + Segment embeddings
│   ├── BERT                  # Main encoder with MLM/NSP heads
│   └── BERTForPretraining    # Wrapper with loss computation
│
├── data_preprocessing.py      # Dataset and data loading
│   ├── WikiTextDataset       # Custom dataset class
│   └── create_dataloaders()  # DataLoader creation
│
├── bert_training.py           # Training script
│   ├── train_epoch()         # Single epoch training
│   ├── evaluate()            # Validation evaluation
│   └── train_bert()          # Main training loop
│
├── bert_eval.py               # Evaluation and demonstration
│   ├── demonstrate_mlm()     # MLM predictions
│   ├── demonstrate_nsp()     # NSP predictions
│   └── evaluate_on_validation() # Metrics computation
│
└── checkpoints/               # Model checkpoints (created during training)
    ├── best_model.pt
    ├── final_model.pt
    ├── training_history.json
    └── training_curves.png
```

## Results

### Training Performance
After 30 epochs of training on WikiText-2, the model achieves:

- **Final Training Loss**: 4.8063
  - MLM Loss: 4.3455
  - NSP Loss: 0.4608
- **Final Validation Loss**: 4.3613
  - MLM Loss: 3.8353
  - NSP Loss: 0.5260
- **MLM Accuracy**: 44.43% (top-1 prediction on masked tokens)
- **NSP Accuracy**: 74.30% (binary classification)

<img width="930" height="616" alt="Screenshot 2025-11-09 000420" src="https://github.com/user-attachments/assets/dcf92193-cd6b-45f4-a346-609cf0f3f8d3" />


The model demonstrates meaningful learning of language patterns:
1. **MLM Performance**: 44.43% accuracy represents significant learning compared to random baseline (0.003% for 30K vocabulary), showing the model captures contextual relationships
2. **NSP Performance**: 74.30% accuracy is substantially above random baseline (50%), indicating the model successfully learns sentence-level coherence
3. **Loss Reduction**: Training loss improved from initial ~8-9 to 4.81, and validation loss to 4.36, demonstrating effective learning without severe overfitting
4. **Training Constraints**: Performance is limited by small dataset (WikiText-2: ~2M tokens vs BERT's 3.3B tokens) and reduced architecture (6 layers, 384-dim vs BERT-base's 12 layers, 768-dim)
5. **Prediction Quality**: While not matching full BERT performance, the model shows improvement in contextual understanding compared to smaller configurations

### Example Predictions

#### Masked Language Modeling
```
Input: "The capital of France is [MASK]."
Top 5 predictions:
  1. unknown     (score: 6.30)
  2. named       (score: 5.32)
  3. built       (score: 5.02)
  4. located     (score: 5.01)
  5. known       (score: 5.00)

Input: "Albert Einstein was a famous [MASK]."
Top 5 predictions:
  1. role        (score: 5.11)
  2. man         (score: 5.08)
  3. film        (score: 4.97)
  4. .           (score: 4.62)
  5. woman       (score: 4.61)

Input: "I love to [MASK] books in my free time."
Top 5 predictions:
  1. be          (score: 7.63)
  2. the         (score: 7.50)
  3. a           (score: 6.41)
  4. my          (score: 6.15)
  5. his         (score: 5.96)
```

**Note**: The model's predictions are not semantically accurate due to training constraints. However, the implementation correctly follows BERT's architecture and training procedure. With more data, longer training, and a larger model, performance would improve significantly.

## Implementation Details

### Key Design Decisions

1. **Scaled Architecture**: Reduced model size for faster training while maintaining architectural principles
2. **Batch First**: Used `batch_first=True` in TransformerEncoder for cleaner code
3. **Efficient Masking**: Pre-computed masked positions to avoid dynamic masking overhead
4. **Gradient Clipping**: Stabilizes training, especially in early epochs
5. **Warmup Schedule**: Linear warmup prevents unstable early training

### Special Tokens
- `[CLS]`: Classification token (ID: 101)
- `[SEP]`: Separator token (ID: 102)
- `[MASK]`: Mask token (ID: 103)
- `[PAD]`: Padding token (ID: 0)

### Data Format
Each training example contains:
- `input_ids`: Tokenized sequence with special tokens
- `segment_ids`: 0 for sentence A, 1 for sentence B
- `attention_mask`: 1 for real tokens, 0 for padding
- `masked_positions`: Indices of masked tokens (padded to 20)
- `masked_labels`: Original token IDs at masked positions
- `is_next`: Binary NSP label (0=random, 1=next)

## Limitations

- **Small Dataset**: WikiText-2 (~2M tokens) vs original BERT (~3.3B tokens)
- **Reduced Architecture**: Fewer layers/dimensions than BERT-base
- **Short Training**: 30 epochs vs original BERT's extensive pre-training
- **Limited Vocabulary**: Uses only uncased vocabulary
- **No Fine-tuning**: Model is only pre-trained, not fine-tuned on downstream tasks

## References

1. **BERT Paper**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

2. **Transformer Paper**: Vaswani, A., et al. (2017). Attention is All You Need. NeurIPS.

## Future Improvements

- Implement whole word masking
- Add SentencePiece tokenization option
- Implement dynamic masking (different masks per epoch)
- Fine-tune on downstream tasks (sentiment analysis, QA)
- Multi-GPU support with DistributedDataParallel
