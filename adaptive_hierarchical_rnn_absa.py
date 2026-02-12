"""
Adaptive Hierarchical RNN for Cross-Domain Aspect-Based Sentiment Analysis (ABSA)

This model:
1. Uses hierarchical RNN (word-level → sentence-level)
2. Focuses on specific aspects (battery, screen, performance, etc.)
3. Adapts across domains (phones → laptops) with minimal retraining
4. Uses meta-learning approach for fast domain adaptation

Architecture:
- Word-Level RNN: Captures local context around aspects
- Sentence-Level RNN: Aggregates information for aspect-focused sentiment
- Aspect Attention: Focuses on relevant parts for each aspect
- Domain Adaptation: Meta-learning for quick adaptation to new domains

Author: Sharan G S  
Date: September 26, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Optional
import pickle
import os

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class AdaptiveHierarchicalRNN(nn.Module):
    """Hierarchical RNN with aspect attention and domain adaptation capabilities"""
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 64, num_aspects: int = 5, 
                 num_classes: int = 3, dropout: float = 0.3):
        super(AdaptiveHierarchicalRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_aspects = num_aspects
        self.num_classes = num_classes  # Negative, Neutral, Positive
        
        # Word embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        # Word-level RNN (bidirectional for better context)
        self.word_rnn = nn.GRU(embedding_dim, hidden_dim, 
                              batch_first=True, bidirectional=True)
        
        # Aspect-specific attention mechanisms
        self.aspect_attentions = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(num_aspects)
        ])
        
        # Sentence-level RNN for each aspect
        self.sentence_rnns = nn.ModuleList([
            nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
            for _ in range(num_aspects)
        ])
        
        # Domain adaptation layers
        self.domain_adapters = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_aspects)
        ])
        
        # Classification heads for each aspect
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            ) for _ in range(num_aspects)
        ])
        
        # Meta-learning parameters for domain adaptation
        self.meta_lr = nn.Parameter(torch.ones(num_aspects) * 0.01)
        
    def forward(self, input_ids: torch.Tensor, aspect_id: int, 
                domain_adapt: bool = False) -> torch.Tensor:
        """
        Forward pass for specific aspect sentiment classification
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            aspect_id: Which aspect to analyze (0-4)
            domain_adapt: Whether to apply domain adaptation
        
        Returns:
            Sentiment logits for the specified aspect
        """
        batch_size, seq_len = input_ids.shape
        
        # Word-level processing
        embedded = self.embedding(input_ids)  # [B, S, E]
        embedded = self.dropout(embedded)
        
        # Word-level RNN
        word_outputs, _ = self.word_rnn(embedded)  # [B, S, H*2]
        
        # Aspect-specific attention
        attention_weights = torch.tanh(self.aspect_attentions[aspect_id](word_outputs))
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # [B, S]
        
        # Weighted aggregation for aspect focus
        aspect_context = torch.sum(word_outputs * attention_weights.unsqueeze(-1), dim=1)  # [B, H*2]
        
        # Sentence-level RNN for aspect
        aspect_context = aspect_context.unsqueeze(1)  # [B, 1, H*2]
        sentence_output, _ = self.sentence_rnns[aspect_id](aspect_context)  # [B, 1, H]
        sentence_output = sentence_output.squeeze(1)  # [B, H]
        
        # Domain adaptation
        if domain_adapt:
            sentence_output = torch.tanh(self.domain_adapters[aspect_id](sentence_output))
        
        # Classification
        logits = self.classifiers[aspect_id](sentence_output)
        
        return logits, attention_weights
    
    def meta_update(self, support_data: List, query_data: List, aspect_id: int):
        """Meta-learning update for domain adaptation"""
        # Save original parameters
        original_params = {}
        for name, param in self.named_parameters():
            if f'aspect_{aspect_id}' in name or 'domain_adapters' in name:
                original_params[name] = param.clone()
        
        # Inner loop: adapt on support set
        support_loss = 0
        for input_ids, labels in support_data:
            logits, _ = self.forward(input_ids, aspect_id, domain_adapt=True)
            loss = F.cross_entropy(logits, labels)
            support_loss += loss
        
        # Gradient step with meta learning rate
        support_loss.backward(retain_graph=True)
        with torch.no_grad():
            for name, param in self.named_parameters():
                if f'aspect_{aspect_id}' in name or 'domain_adapters' in name:
                    if param.grad is not None:
                        param.data -= self.meta_lr[aspect_id] * param.grad
        
        # Outer loop: evaluate on query set
        query_loss = 0
        for input_ids, labels in query_data:
            logits, _ = self.forward(input_ids, aspect_id, domain_adapt=True)
            loss = F.cross_entropy(logits, labels)
            query_loss += loss
        
        return query_loss

class ABSADataGenerator:
    """Generate synthetic ABSA data for different domains"""
    
    def __init__(self):
        # Define aspects
        self.aspects = {
            0: 'battery',
            1: 'screen', 
            2: 'performance',
            3: 'design',
            4: 'price'
        }
        
        # Domain-specific vocabularies
        self.domain_vocabs = {
            'phone': {
                'battery': ['battery', 'charge', 'power', 'drain', 'lasting', 'life'],
                'screen': ['display', 'screen', 'bright', 'clear', 'resolution', 'pixels'],
                'performance': ['fast', 'slow', 'lag', 'smooth', 'responsive', 'speed'],
                'design': ['beautiful', 'sleek', 'ugly', 'stylish', 'premium', 'cheap'],
                'price': ['expensive', 'cheap', 'worth', 'overpriced', 'value', 'cost']
            },
            'laptop': {
                'battery': ['battery', 'power', 'charging', 'unplugged', 'portable', 'runtime'],
                'screen': ['monitor', 'display', 'screen', 'panel', 'brightness', 'colors'],
                'performance': ['processor', 'RAM', 'graphics', 'gaming', 'multitasking', 'benchmark'],
                'design': ['build', 'keyboard', 'trackpad', 'ports', 'weight', 'thickness'],
                'price': ['budget', 'premium', 'affordable', 'investment', 'deals', 'money']
            }
        }
        
        # Sentiment templates
        self.sentiment_templates = {
            0: [  # Negative
                "The {aspect} is terrible and {negative_word}",
                "Really disappointed with the {aspect}, it's {negative_word}",
                "The {aspect} {negative_word} and doesn't work well",
                "Poor {aspect} quality, very {negative_word}"
            ],
            1: [  # Neutral
                "The {aspect} is okay, nothing special about {neutral_word}",
                "Average {aspect}, {neutral_word} performance",
                "The {aspect} is decent, {neutral_word} overall",
                "Standard {aspect}, meets {neutral_word} expectations"
            ],
            2: [  # Positive
                "The {aspect} is amazing and very {positive_word}",
                "Love the {aspect}, it's so {positive_word}",
                "Excellent {aspect} with {positive_word} quality",
                "The {aspect} {positive_word} and works perfectly"
            ]
        }
        
        self.negative_words = ['awful', 'bad', 'poor', 'disappointing', 'useless']
        self.neutral_words = ['average', 'standard', 'normal', 'typical', 'basic']
        self.positive_words = ['great', 'excellent', 'amazing', 'fantastic', 'outstanding']
        
        # Build vocabulary
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self._build_vocabulary()
        
    def _build_vocabulary(self):
        """Build vocabulary from all domains and templates"""
        vocab_words = set()
        
        # Add domain-specific words
        for domain_vocab in self.domain_vocabs.values():
            for aspect_words in domain_vocab.values():
                vocab_words.update(aspect_words)
        
        # Add sentiment words
        vocab_words.update(self.negative_words + self.neutral_words + self.positive_words)
        
        # Add common words
        common_words = ['the', 'is', 'and', 'very', 'it', 'with', 'really', 'about',
                       'quality', 'work', 'good', 'well', 'like', 'love', 'hate',
                       'works', 'doesn', 'performance', 'overall', 'meets']
        vocab_words.update(common_words)
        
        # Create word mappings
        for word in vocab_words:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
    
    def generate_sample(self, domain: str, aspect_id: int, sentiment: int) -> Tuple[str, List[int]]:
        """Generate a single ABSA sample"""
        aspect_name = self.aspects[aspect_id]
        
        # Choose template and words based on sentiment
        template = random.choice(self.sentiment_templates[sentiment])
        
        if sentiment == 0:  # Negative
            sentiment_word = random.choice(self.negative_words)
            template = template.format(aspect=aspect_name, negative_word=sentiment_word)
        elif sentiment == 1:  # Neutral
            sentiment_word = random.choice(self.neutral_words)
            template = template.format(aspect=aspect_name, neutral_word=sentiment_word)
        else:  # Positive
            sentiment_word = random.choice(self.positive_words)
            template = template.format(aspect=aspect_name, positive_word=sentiment_word)
        
        # Add domain-specific aspect words
        if random.random() > 0.5:
            aspect_word = random.choice(self.domain_vocabs[domain][aspect_name])
            template = template.replace(aspect_name, aspect_word, 1)
        
        # Convert to token IDs
        words = template.lower().replace(',', '').replace('.', '').split()
        token_ids = []
        for word in words:
            token_ids.append(self.word_to_idx.get(word, 1))  # 1 = <UNK>
        
        return template, token_ids
    
    def generate_dataset(self, domain: str, samples_per_aspect: int = 200) -> Tuple[List, List, List]:
        """Generate dataset for a domain"""
        texts, token_sequences, labels, aspects = [], [], [], []
        
        for aspect_id in range(5):  # 5 aspects
            for sentiment in range(3):  # 3 sentiment classes
                for _ in range(samples_per_aspect // 3):
                    text, tokens = self.generate_sample(domain, aspect_id, sentiment)
                    texts.append(text)
                    token_sequences.append(tokens)
                    labels.append(sentiment)
                    aspects.append(aspect_id)
        
        return texts, token_sequences, labels, aspects

def pad_sequences(sequences: List[List[int]], max_len: int = 32) -> torch.Tensor:
    """Pad sequences to same length"""
    padded = []
    for seq in sequences:
        if len(seq) > max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [0] * (max_len - len(seq)))
    return torch.LongTensor(padded)

def train_model(model: AdaptiveHierarchicalRNN, train_data: Tuple, 
                val_data: Tuple, epochs: int = 50) -> List[float]:
    """Train the ABSA model"""
    texts, token_sequences, labels, aspects = train_data
    val_texts, val_token_sequences, val_labels, val_aspects = val_data
    
    # Prepare data
    train_tokens = pad_sequences(token_sequences)
    train_labels = torch.LongTensor(labels)
    train_aspects = torch.LongTensor(aspects)
    
    val_tokens = pad_sequences(val_token_sequences)
    val_labels = torch.LongTensor(val_labels)
    val_aspects = torch.LongTensor(val_aspects)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    
    print("Training Hierarchical RNN for ABSA...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Shuffle data
        indices = list(range(len(train_tokens)))
        random.shuffle(indices)
        
        batch_size = 32
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batch_tokens = train_tokens[batch_indices]
            batch_labels = train_labels[batch_indices]
            batch_aspects = train_aspects[batch_indices]
            
            optimizer.zero_grad()
            
            # Process each aspect separately in the batch
            batch_loss = 0
            for j, (tokens, label, aspect_id) in enumerate(zip(batch_tokens, batch_labels, batch_aspects)):
                logits, attention = model(tokens.unsqueeze(0), aspect_id.item())
                loss = criterion(logits, label.unsqueeze(0))
                batch_loss += loss
            
            batch_loss = batch_loss / len(batch_tokens)
            batch_loss.backward()
            optimizer.step()
            
            total_loss += batch_loss.item()
        
        avg_loss = total_loss / (len(train_tokens) // batch_size)
        train_losses.append(avg_loss)
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_accuracy = evaluate_model(model, val_data)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_accuracy:.3f}")
    
    return train_losses

def evaluate_model(model: AdaptiveHierarchicalRNN, test_data: Tuple) -> float:
    """Evaluate model performance"""
    texts, token_sequences, labels, aspects = test_data
    
    test_tokens = pad_sequences(token_sequences)
    test_labels = torch.LongTensor(labels)
    test_aspects = torch.LongTensor(aspects)
    
    model.eval()
    correct = 0
    total = 0
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for tokens, label, aspect_id in zip(test_tokens, test_labels, test_aspects):
            logits, attention = model(tokens.unsqueeze(0), aspect_id.item())
            predicted = torch.argmax(logits, dim=1)
            
            predictions.append(predicted.item())
            true_labels.append(label.item())
            
            if predicted.item() == label.item():
                correct += 1
            total += 1
    
    accuracy = correct / total
    return accuracy

def demonstrate_attention(model: AdaptiveHierarchicalRNN, data_generator: ABSADataGenerator,
                         sample_text: str, aspect_id: int):
    """Visualize attention weights for interpretability"""
    # Tokenize sample
    words = sample_text.lower().replace(',', '').replace('.', '').split()
    token_ids = [data_generator.word_to_idx.get(word, 1) for word in words]
    
    # Pad and convert to tensor
    padded_tokens = pad_sequences([token_ids], max_len=len(token_ids) + 5)
    
    model.eval()
    with torch.no_grad():
        logits, attention_weights = model(padded_tokens, aspect_id)
        predicted_sentiment = torch.argmax(logits, dim=1).item()
    
    # Visualize attention
    plt.figure(figsize=(12, 6))
    
    # Attention heatmap
    plt.subplot(1, 2, 1)
    attention_np = attention_weights[0, :len(words)].numpy()
    plt.bar(range(len(words)), attention_np)
    plt.xticks(range(len(words)), words, rotation=45)
    plt.title(f'Attention Weights for {data_generator.aspects[aspect_id].title()} Aspect')
    plt.ylabel('Attention Weight')
    
    # Prediction
    plt.subplot(1, 2, 2)
    sentiment_names = ['Negative', 'Neutral', 'Positive']
    probs = F.softmax(logits, dim=1)[0].numpy()
    colors = ['red', 'gray', 'green']
    bars = plt.bar(sentiment_names, probs, color=colors, alpha=0.7)
    
    # Highlight predicted class
    bars[predicted_sentiment].set_alpha(1.0)
    bars[predicted_sentiment].set_edgecolor('black')
    bars[predicted_sentiment].set_linewidth(2)
    
    plt.title(f'Sentiment Prediction: {sentiment_names[predicted_sentiment]}')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    plt.show()

def demonstrate_domain_adaptation(model: AdaptiveHierarchicalRNN, 
                                data_generator: ABSADataGenerator):
    """Demonstrate cross-domain adaptation capability"""
    print("\n🔄 DEMONSTRATING DOMAIN ADAPTATION")
    print("=" * 50)
    
    # Generate source domain data (phones)
    print("📱 Training on Phone Reviews...")
    phone_texts, phone_tokens, phone_labels, phone_aspects = data_generator.generate_dataset('phone', 300)
    
    # Split into train/val
    train_size = int(0.8 * len(phone_tokens))
    train_data = (phone_texts[:train_size], phone_tokens[:train_size], 
                  phone_labels[:train_size], phone_aspects[:train_size])
    val_data = (phone_texts[train_size:], phone_tokens[train_size:], 
                phone_labels[train_size:], phone_aspects[train_size:])
    
    # Train on phone domain
    train_losses = train_model(model, train_data, val_data, epochs=30)
    phone_accuracy = evaluate_model(model, val_data)
    print(f"✅ Phone Domain Accuracy: {phone_accuracy:.3f}")
    
    # Test on laptop domain (without adaptation)
    print("\n💻 Testing on Laptop Reviews (No Adaptation)...")
    laptop_texts, laptop_tokens, laptop_labels, laptop_aspects = data_generator.generate_dataset('laptop', 100)
    laptop_test_data = (laptop_texts, laptop_tokens, laptop_labels, laptop_aspects)
    
    laptop_accuracy_no_adapt = evaluate_model(model, laptop_test_data)
    print(f"❌ Laptop Domain Accuracy (No Adaptation): {laptop_accuracy_no_adapt:.3f}")
    
    # Few-shot adaptation to laptop domain
    print("\n🎯 Few-shot Adaptation to Laptop Domain...")
    adaptation_samples = 50  # Only 50 samples for adaptation
    adapt_data = (laptop_texts[:adaptation_samples], laptop_tokens[:adaptation_samples],
                  laptop_labels[:adaptation_samples], laptop_aspects[:adaptation_samples])
    
    # Meta-learning adaptation (simplified)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate
    criterion = nn.CrossEntropyLoss()
    
    # Quick adaptation
    model.train()
    for epoch in range(10):  # Few epochs
        total_loss = 0
        for i, (tokens, label, aspect_id) in enumerate(zip(adapt_data[1], adapt_data[2], adapt_data[3])):
            if i >= adaptation_samples:
                break
                
            optimizer.zero_grad()
            padded_tokens = pad_sequences([tokens])
            logits, _ = model(padded_tokens, aspect_id, domain_adapt=True)
            loss = criterion(logits, torch.LongTensor([label]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Test after adaptation
    laptop_accuracy_adapted = evaluate_model(model, laptop_test_data)
    print(f"✅ Laptop Domain Accuracy (After Adaptation): {laptop_accuracy_adapted:.3f}")
    print(f"📈 Improvement: {laptop_accuracy_adapted - laptop_accuracy_no_adapt:.3f}")
    
    return phone_accuracy, laptop_accuracy_no_adapt, laptop_accuracy_adapted

def main():
    """Main function to demonstrate Adaptive Hierarchical RNN for ABSA"""
    print("🧠 ADAPTIVE HIERARCHICAL RNN FOR CROSS-DOMAIN ABSA")
    print("=" * 60)
    
    # Initialize data generator
    print("\n📊 Initializing ABSA Data Generator...")
    data_generator = ABSADataGenerator()
    print(f"✅ Vocabulary Size: {len(data_generator.word_to_idx)}")
    print(f"✅ Aspects: {list(data_generator.aspects.values())}")
    print(f"✅ Domains: phone, laptop")
    
    # Create model
    print("\n🏗️ Building Adaptive Hierarchical RNN...")
    model = AdaptiveHierarchicalRNN(
        vocab_size=len(data_generator.word_to_idx),
        embedding_dim=128,
        hidden_dim=64,
        num_aspects=5,
        num_classes=3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Demonstrate domain adaptation
    phone_acc, laptop_no_adapt, laptop_adapted = demonstrate_domain_adaptation(model, data_generator)
    
    # Demonstrate attention mechanism
    print("\n🔍 DEMONSTRATING ATTENTION MECHANISM")
    print("=" * 50)
    
    sample_reviews = [
        ("The battery life is amazing and lasts all day", 0),  # battery aspect
        ("The screen display is crystal clear and bright", 1),  # screen aspect  
        ("Performance is terrible, very slow and laggy", 2),   # performance aspect
        ("Beautiful design with premium build quality", 3),    # design aspect
        ("Too expensive for what you get, overpriced", 4)      # price aspect
    ]
    
    for text, aspect_id in sample_reviews:
        print(f"\n📝 Review: '{text}'")
        print(f"🎯 Analyzing {data_generator.aspects[aspect_id].title()} aspect...")
        demonstrate_attention(model, data_generator, text, aspect_id)
        input("Press Enter to continue to next example...")
    
    # Save model
    model_path = "/Users/sharan/TEST/adaptive_hierarchical_rnn_absa.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': len(data_generator.word_to_idx),
        'word_to_idx': data_generator.word_to_idx,
        'idx_to_word': data_generator.idx_to_word,
        'aspects': data_generator.aspects
    }, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Results summary
    print("\n🎉 ADAPTIVE HIERARCHICAL RNN RESULTS")
    print("=" * 45)
    print(f"📱 Phone Domain Accuracy: {phone_acc:.1%}")
    print(f"💻 Laptop Domain (No Adapt): {laptop_no_adapt:.1%}")  
    print(f"🎯 Laptop Domain (Adapted): {laptop_adapted:.1%}")
    print(f"📈 Domain Adaptation Gain: +{(laptop_adapted - laptop_no_adapt):.1%}")
    
    print("\n🔬 KEY FEATURES DEMONSTRATED:")
    print("   • 📝 Aspect-Based Sentiment Analysis")
    print("   • 🏗️ Hierarchical RNN (Word → Sentence)")
    print("   • 🎯 Aspect-Specific Attention")
    print("   • 🔄 Cross-Domain Adaptation")
    print("   • 🧠 Meta-Learning Approach")
    print("   • 📊 Interpretable Attention Weights")
    
    print("\n💼 BUSINESS APPLICATIONS:")
    print("   • 🛍️ Product Review Analysis")
    print("   • 📱 Customer Feedback Mining")
    print("   • 🎯 Aspect-Specific Insights")
    print("   • 🔄 Multi-Domain Deployment")
    print("   • ⚡ Fast Domain Adaptation")

if __name__ == "__main__":
    main()