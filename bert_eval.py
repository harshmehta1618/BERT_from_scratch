import torch
import torch.nn.functional as F
from transformers import BertTokenizer
import json
import os


def load_trained_model(checkpoint_path='checkpoints/best_model.pt', device='cuda'):
    """Load a trained BERT model"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    bert = BERT(**config)
    model = BERTForPretraining(bert)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded (trained for {checkpoint['epoch']} epochs)")
    print(f"✓ Validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, config


def demonstrate_mlm(model, tokenizer, device, sentences=None):
    """Demonstrate Masked Language Modeling"""
    
    if sentences is None:
        sentences = [
            "The capital of France is [MASK].",
            "Albert Einstein was a famous [MASK].",
            "I love to [MASK] books in my free time.",
            "The [MASK] is the largest planet in our solar system.",
            "Machine [MASK] is a subset of artificial intelligence.",
            "Python is a popular programming [MASK].",
            "The [MASK] revolves around the Earth.",
            "Water boils at [MASK] degrees Celsius.",
            "The Amazon is the largest [MASK] in the world.",
            "Shakespeare wrote many famous [MASK]."
        ]
    
    print("\n" + "="*80)
    print("MASKED LANGUAGE MODELING DEMONSTRATION")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        for i, sentence in enumerate(sentences, 1):
            print(f"\n{'─'*80}")
            print(f"Example {i}:")
            print(f"Input: {sentence}")
            
            # Tokenize
            tokens = tokenizer.tokenize(sentence)
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # Check if [MASK] exists
            if tokenizer.mask_token_id not in input_ids:
                print("⚠ No [MASK] token found. Skipping...")
                continue
            
            # Find mask position
            mask_pos = input_ids.index(tokenizer.mask_token_id)
            
            # Prepare input
            input_tensor = torch.tensor([input_ids]).to(device)
            segment_tensor = torch.zeros_like(input_tensor).to(device)
            
            # Get predictions
            mlm_output, _ = model.bert(input_tensor, segment_tensor)
            
            # Get top predictions for masked position
            mask_predictions = mlm_output[0, mask_pos, :]
            probs = F.softmax(mask_predictions, dim=0)
            top_k = torch.topk(probs, k=10)
            
            print(f"\nTop 10 Predictions:")
            print(f"{'Rank':<6} {'Token':<20} {'Probability':<12} {'Score':<10}")
            print("─" * 50)
            
            for rank, (prob, token_id) in enumerate(zip(top_k.values, top_k.indices), 1):
                token = tokenizer.convert_ids_to_tokens([token_id.item()])[0]
                print(f"{rank:<6} {token:<20} {prob.item():.2%}        {mask_predictions[token_id].item():.2f}")
            
            # Show the sentence with top prediction
            top_token = tokenizer.convert_ids_to_tokens([top_k.indices[0].item()])[0]
            completed = sentence.replace('[MASK]', f'**{top_token}**')
            print(f"\nCompleted: {completed}")


def demonstrate_nsp(model, tokenizer, device):
    """Demonstrate Next Sentence Prediction"""
    
    # Define sentence pairs (sentence_a, sentence_b, is_actually_next)
    pairs = [
        ("The weather is nice today.", "I think I'll go for a walk.", True),
        ("The weather is nice today.", "The capital of France is Paris.", False),
        ("Machine learning is a subset of AI.", "It uses algorithms to learn from data.", True),
        ("Machine learning is a subset of AI.", "Pizza is a popular food in Italy.", False),
        ("The cat sat on the mat.", "It was sleeping peacefully.", True),
        ("The cat sat on the mat.", "Quantum physics is fascinating.", False),
        ("Python is a programming language.", "It is widely used for data science.", True),
        ("Python is a programming language.", "The Moon orbits the Earth.", False),
    ]
    
    print("\n" + "="*80)
    print("NEXT SENTENCE PREDICTION DEMONSTRATION")
    print("="*80)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (sent_a, sent_b, is_next) in enumerate(pairs, 1):
            print(f"\n{'─'*80}")
            print(f"Pair {i}:")
            print(f"Sentence A: {sent_a}")
            print(f"Sentence B: {sent_b}")
            print(f"Actual: {'Next Sentence' if is_next else 'Random Sentence'}")
            
            # Tokenize
            tokens_a = tokenizer.tokenize(sent_a)
            tokens_b = tokenizer.tokenize(sent_b)
            
            # Create input: [CLS] + sent_a + [SEP] + sent_b + [SEP]
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            
            # Pad to same length
            max_len = 64
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_len - len(input_ids))
            segment_ids = segment_ids + [0] * (max_len - len(segment_ids))
            
            # Convert to tensors
            input_tensor = torch.tensor([input_ids[:max_len]]).to(device)
            segment_tensor = torch.tensor([segment_ids[:max_len]]).to(device)
            
            # Get prediction
            _, nsp_output = model.bert(input_tensor, segment_tensor)
            probs = F.softmax(nsp_output, dim=1)
            prediction = torch.argmax(nsp_output, dim=1).item()
            
            # Display results
            print(f"\nPrediction Probabilities:")
            print(f"  Random Sentence: {probs[0, 0].item():.2%}")
            print(f"  Next Sentence:   {probs[0, 1].item():.2%}")
            print(f"\nPredicted: {'Next Sentence' if prediction == 1 else 'Random Sentence'}")
            
            # Check if correct
            actual_label = 1 if is_next else 0
            is_correct = (prediction == actual_label)
            print(f"Result: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            
            if is_correct:
                correct += 1
            total += 1
    
    accuracy = correct / total
    print(f"\n{'='*80}")
    print(f"NSP Accuracy on Examples: {correct}/{total} = {accuracy:.2%}")
    print(f"{'='*80}")


def analyze_attention(model, tokenizer, device, sentence="The cat sat on the mat."):
    """Analyze attention patterns (optional advanced feature)"""
    print("\n" + "="*80)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing sentence: {sentence}")
    
    # Tokenize
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    print(f"Tokens: {' '.join(tokens)}")
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]).to(device)
    segment_tensor = torch.zeros_like(input_tensor).to(device)
    
    model.eval()
    with torch.no_grad():
        # Get embeddings and pass through encoder
        embedded = model.bert.embedding(input_tensor, segment_tensor)
        
        # Note: Extracting attention weights requires modifying the forward pass
        # For now, we just show that the model processes the input
        encoded = model.bert.encoder_block(embedded)
        
        print(f"\nEmbedding shape: {embedded.shape}")
        print(f"Encoded output shape: {encoded.shape}")
        print("Successfully processed through all layers")


def evaluate_on_validation(model, device):
    """Evaluate on validation set"""
    print("\n" + "="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    
    # Load validation data
    _, val_loader, _ = create_dataloaders(batch_size=32, max_len=128)
    
    model.eval()
    total_loss = 0
    correct_nsp = 0
    total_nsp = 0
    correct_mlm = 0
    total_mlm = 0
    
    print("Evaluating on validation set...")
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            masked_positions = batch['masked_positions'].to(device)
            masked_labels = batch['masked_labels'].to(device)
            is_next = batch['is_next'].to(device)
            
            # Forward pass
            loss, mlm_loss, nsp_loss = model(
                input_ids, segment_ids, masked_positions,
                masked_labels, is_next, attention_mask
            )
            
            # Get predictions
            mlm_output, nsp_output = model.bert(input_ids, segment_ids, attention_mask)
            
            # NSP accuracy
            nsp_pred = torch.argmax(nsp_output, dim=1)
            correct_nsp += (nsp_pred == is_next).sum().item()
            total_nsp += is_next.size(0)
            
            # MLM accuracy
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
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    nsp_accuracy = correct_nsp / total_nsp
    mlm_accuracy = correct_mlm / total_mlm
    
    print(f"\nValidation Results:")
    print(f"{'─'*50}")
    print(f"Total Loss:       {avg_loss:.4f}")
    print(f"MLM Accuracy:     {mlm_accuracy:.2%} ({correct_mlm}/{total_mlm})")
    print(f"NSP Accuracy:     {nsp_accuracy:.2%} ({correct_nsp}/{total_nsp})")
    print(f"{'─'*50}")
    
    return avg_loss, mlm_accuracy, nsp_accuracy


def main():
    """Main evaluation and demonstration script"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model, config = load_trained_model(device=device)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("\n" + "="*80)
    print("BERT MODEL EVALUATION AND DEMONSTRATION")
    print("="*80)
    print("\nWhat would you like to do?")
    print("1. Masked Language Modeling Demo")
    print("2. Next Sentence Prediction Demo")
    print("3. Full Validation Evaluation")
    print("4. All of the above")
    print("5. Custom sentence MLM")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1' or choice == '4':
        demonstrate_mlm(model, tokenizer, device)
    
    if choice == '2' or choice == '4':
        demonstrate_nsp(model, tokenizer, device)
    
    if choice == '3' or choice == '4':
        evaluate_on_validation(model, device)
    
    if choice == '5':
        print("\nEnter sentences with [MASK] tokens (press Enter twice to finish):")
        sentences = []
        while True:
            sent = input()
            if not sent:
                break
            sentences.append(sent)
        if sentences:
            demonstrate_mlm(model, tokenizer, device, sentences)
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)


if __name__ == "__main__":
    main()