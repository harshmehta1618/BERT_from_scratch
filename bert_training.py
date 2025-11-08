from bert_model import BERT, BERTForPretraining
from data_preprocessing import create_dataloaders
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup  # Fixed import
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime




def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        masked_positions = batch['masked_positions'].to(device)
        masked_labels = batch['masked_labels'].to(device)
        is_next = batch['is_next'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss, mlm_loss, nsp_loss = model(
            input_ids,
            segment_ids,
            masked_positions,
            masked_labels,
            is_next,
            attention_mask
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'mlm': mlm_loss.item(),
            'nsp': nsp_loss.item(),
            'lr': scheduler.get_last_lr()[0]
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_mlm_loss = total_mlm_loss / len(train_loader)
    avg_nsp_loss = total_nsp_loss / len(train_loader)
    
    return avg_loss, avg_mlm_loss, avg_nsp_loss


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    
    # For NSP accuracy
    correct_nsp = 0
    total_nsp = 0
    
    # For MLM accuracy
    correct_mlm = 0
    total_mlm = 0
    
    for batch in tqdm(val_loader, desc='Evaluating'):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        masked_positions = batch['masked_positions'].to(device)
        masked_labels = batch['masked_labels'].to(device)
        is_next = batch['is_next'].to(device)
        
        # Forward pass
        loss, mlm_loss, nsp_loss = model(
            input_ids,
            segment_ids,
            masked_positions,
            masked_labels,
            is_next,
            attention_mask
        )
        
        # Get predictions
        mlm_output, nsp_output = model.bert(input_ids, segment_ids, attention_mask)
        
        # NSP accuracy
        nsp_pred = torch.argmax(nsp_output, dim=1)
        correct_nsp += (nsp_pred == is_next).sum().item()
        total_nsp += is_next.size(0)
        
        # MLM accuracy (only on masked positions)
        batch_size, seq_len, vocab_size = mlm_output.size()
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand_as(masked_positions)
        flat_masked_pos = (batch_indices * seq_len + masked_positions).view(-1)
        
        mlm_output_flat = mlm_output.view(-1, vocab_size)
        masked_predictions = mlm_output_flat[flat_masked_pos]
        mlm_pred = torch.argmax(masked_predictions, dim=1)
        
        masked_labels_flat = masked_labels.view(-1)
        valid_mask = (masked_labels_flat != -1)
        
        correct_mlm += ((mlm_pred == masked_labels_flat) & valid_mask).sum().item()
        total_mlm += valid_mask.sum().item()
        
        # Accumulate losses
        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_mlm_loss = total_mlm_loss / len(val_loader)
    avg_nsp_loss = total_nsp_loss / len(val_loader)
    nsp_accuracy = correct_nsp / total_nsp
    mlm_accuracy = correct_mlm / total_mlm
    
    return avg_loss, avg_mlm_loss, avg_nsp_loss, nsp_accuracy, mlm_accuracy


def demonstrate_predictions(model, tokenizer, device, num_examples=3):
    """Show some example predictions"""
    model.eval()
    
    print("\n" + "="*80)
    print("DEMONSTRATION: Masked Language Model Predictions")
    print("="*80)
    
    # Example sentences
    examples = [
        "The capital of France is [MASK].",
        "Albert Einstein was a famous [MASK].",
        "I love to [MASK] books in my free time."
    ]
    
    with torch.no_grad():
        for i, sentence in enumerate(examples[:num_examples]):
            print(f"\nExample {i+1}: {sentence}")
            
            # Tokenize
            tokens = tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Find mask position
            mask_pos = input_ids.index(tokenizer.mask_token_id)
            
            # Prepare input
            input_tensor = torch.tensor([input_ids]).to(device)
            segment_tensor = torch.zeros_like(input_tensor).to(device)
            
            # Get predictions
            mlm_output, _ = model.bert(input_tensor, segment_tensor)
            
            # Get top 5 predictions for masked position
            mask_predictions = mlm_output[0, mask_pos, :]
            top_k = torch.topk(mask_predictions, k=5)
            
            print(f"Top 5 predictions:")
            for rank, (score, token_id) in enumerate(zip(top_k.values, top_k.indices)):
                token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
                print(f"  {rank+1}. {token:15s} (score: {score.item():.2f})")


def plot_training_curves(history, save_path='training_curves.png'):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MLM Loss
    axes[0, 1].plot(history['train_mlm_loss'], label='Train')
    axes[0, 1].plot(history['val_mlm_loss'], label='Validation')
    axes[0, 1].set_title('MLM Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # NSP Loss
    axes[1, 0].plot(history['train_nsp_loss'], label='Train')
    axes[1, 0].plot(history['val_nsp_loss'], label='Validation')
    axes[1, 0].set_title('NSP Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Accuracies
    axes[1, 1].plot(history['val_mlm_accuracy'], label='MLM Accuracy')
    axes[1, 1].plot(history['val_nsp_accuracy'], label='NSP Accuracy')
    axes[1, 1].set_title('Validation Accuracies')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {save_path}")


def train_bert(
    num_epochs=30,
    batch_size=12,
    learning_rate=3e-5,
    max_len=128,
    warmup_steps=1000,
    save_dir='checkpoints'
):
    """Main training function"""
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    train_loader, val_loader, tokenizer = create_dataloaders(
        batch_size=batch_size,
        max_len=max_len
    )
    
    # Model configuration
    vocab_size = tokenizer.vocab_size
    config = {
        'vocab_size': vocab_size,
        'n_segments': 2,
        'max_len': max_len,
        'embed_dim': 384,
        'n_layers': 6,
        'attn_heads': 8,
        'dropout': 0.1
    }
    
    print("\n" + "="*80)
    print("Model Configuration")
    print("="*80)
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Create model
    bert = BERT(**config)
    model = BERTForPretraining(bert)
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_mlm_loss': [],
        'train_nsp_loss': [],
        'val_loss': [],
        'val_mlm_loss': [],
        'val_nsp_loss': [],
        'val_mlm_accuracy': [],
        'val_nsp_accuracy': []
    }
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        
        # Train
        train_loss, train_mlm_loss, train_nsp_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # Evaluate
        val_loss, val_mlm_loss, val_nsp_loss, nsp_acc, mlm_acc = evaluate(
            model, val_loader, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mlm_loss'].append(train_mlm_loss)
        history['train_nsp_loss'].append(train_nsp_loss)
        history['val_loss'].append(val_loss)
        history['val_mlm_loss'].append(val_mlm_loss)
        history['val_nsp_loss'].append(val_nsp_loss)
        history['val_mlm_accuracy'].append(mlm_acc)
        history['val_nsp_accuracy'].append(nsp_acc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} (MLM: {train_mlm_loss:.4f}, NSP: {train_nsp_loss:.4f})")
        print(f"  Val Loss:   {val_loss:.4f} (MLM: {val_mlm_loss:.4f}, NSP: {val_nsp_loss:.4f})")
        print(f"  Val MLM Accuracy: {mlm_acc:.4f}")
        print(f"  Val NSP Accuracy: {nsp_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
    
    # Save final model
    final_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, final_path)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(history, os.path.join(save_dir, 'training_curves.png'))
    
    # Demonstrate predictions
    demonstrate_predictions(model, tokenizer, device)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {save_dir}/")
    
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_bert(
        num_epochs=30,
        batch_size=12,
        learning_rate=3e-5,
        max_len=128,
        warmup_steps=1000
    )
