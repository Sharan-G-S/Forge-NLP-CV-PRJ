"""
Window-based Language Model using Co-occurrence Statistics
This implements a language model that predicts next words based on co-occurrence patterns
within sliding windows of text.
"""

import numpy as np
from collections import defaultdict, Counter
import re
import math
from typing import List, Dict, Tuple, Optional

class WindowBasedLanguageModel:
    """
    A language model based on word co-occurrence within sliding windows.
    Uses co-occurrence statistics to predict the next word given a context.
    """
    
    def __init__(self, window_size: int = 5, smoothing: float = 1.0):
        """
        Initialize the language model.
        
        Args:
            window_size: Size of the sliding window for co-occurrence
            smoothing: Smoothing parameter for probability estimation (Laplace smoothing)
        """
        self.window_size = window_size
        self.smoothing = smoothing
        self.vocab = set()
        self.word_counts = Counter()
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.is_trained = False
        
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text by cleaning and tokenizing.
        
        Args:
            text: Raw input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        
        # Remove punctuation except periods, questions, exclamations
        text = re.sub(r'[^\w\s.!?]', '', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Add start and end tokens
        tokens = ['<START>'] + tokens + ['<END>']
        
        return tokens
    
    def extract_cooccurrences(self, tokens: List[str]) -> None:
        """
        Extract co-occurrence statistics from tokens using sliding windows.
        
        Args:
            tokens: List of preprocessed tokens
        """
        for i in range(len(tokens)):
            # Define window boundaries
            start = max(0, i - self.window_size // 2)
            end = min(len(tokens), i + self.window_size // 2 + 1)
            
            # Current word as target
            target_word = tokens[i]
            
            # Context words in the window (excluding target)
            context_words = []
            for j in range(start, end):
                if j != i:  # Exclude the target word itself
                    context_words.append(tokens[j])
            
            # Update co-occurrence counts
            for context_word in context_words:
                self.cooccurrence_matrix[context_word][target_word] += 1
                self.context_counts[context_word] += 1
            
            # Update vocabulary and word counts
            self.vocab.add(target_word)
            self.word_counts[target_word] += 1
    
    def train(self, texts: List[str]) -> None:
        """
        Train the language model on a list of texts.
        
        Args:
            texts: List of text documents for training
        """
        print(f"Training window-based language model with window size {self.window_size}...")
        
        # Reset model state
        self.vocab = set()
        self.word_counts = Counter()
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        
        # Process each text
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing document {i+1}/{len(texts)}")
            
            tokens = self.preprocess_text(text)
            self.extract_cooccurrences(tokens)
        
        self.is_trained = True
        print(f"Training completed! Vocabulary size: {len(self.vocab)}")
        print(f"Total co-occurrence pairs: {sum(len(targets) for targets in self.cooccurrence_matrix.values())}")
    
    def get_word_probability(self, target_word: str, context_words: List[str]) -> float:
        """
        Calculate probability of target word given context words.
        
        Args:
            target_word: Word to predict
            context_words: List of context words
            
        Returns:
            Probability of target word given context
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        # Calculate co-occurrence score
        cooccur_score = 0
        context_count = 0
        
        for context_word in context_words:
            if context_word in self.cooccurrence_matrix:
                cooccur_score += self.cooccurrence_matrix[context_word][target_word]
                context_count += self.context_counts[context_word]
        
        if context_count == 0:
            # Fallback to unigram probability
            return (self.word_counts[target_word] + self.smoothing) / (sum(self.word_counts.values()) + self.smoothing * len(self.vocab))
        
        # Apply smoothing
        numerator = cooccur_score + self.smoothing
        denominator = context_count + self.smoothing * len(self.vocab)
        
        return numerator / denominator
    
    def predict_next_words(self, context: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the most likely next words given a context.
        
        Args:
            context: Context string
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions!")
        
        context_tokens = self.preprocess_text(context)
        
        # Use last few words as context (up to window size)
        context_words = context_tokens[-min(len(context_tokens), self.window_size-1):]
        
        # Calculate probabilities for all vocabulary words
        word_probs = []
        for word in self.vocab:
            if word not in ['<START>', '<END>']:  # Exclude special tokens from predictions
                prob = self.get_word_probability(word, context_words)
                word_probs.append((word, prob))
        
        # Sort by probability and return top k
        word_probs.sort(key=lambda x: x[1], reverse=True)
        return word_probs[:top_k]
    
    def generate_text(self, seed_text: str, max_length: int = 50) -> str:
        """
        Generate text by iteratively predicting next words.
        
        Args:
            seed_text: Initial text to start generation
            max_length: Maximum number of words to generate
            
        Returns:
            Generated text string
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before generating text!")
        
        generated_tokens = self.preprocess_text(seed_text)[:-1]  # Remove <END> token
        
        for _ in range(max_length):
            # Get context for prediction
            context_words = generated_tokens[-min(len(generated_tokens), self.window_size-1):]
            
            # Get next word probabilities
            word_probs = []
            for word in self.vocab:
                if word not in ['<START>', '<END>']:
                    prob = self.get_word_probability(word, context_words)
                    word_probs.append((word, prob))
            
            if not word_probs:
                break
            
            # Sample next word based on probabilities
            word_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Use weighted random sampling from top candidates
            top_candidates = word_probs[:10]  # Consider top 10 candidates
            weights = [prob for _, prob in top_candidates]
            
            if sum(weights) == 0:
                break
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Sample word
            next_word = np.random.choice([word for word, _ in top_candidates], p=weights)
            
            if next_word == '<END>':
                break
            
            generated_tokens.append(next_word)
        
        # Convert back to text (remove special tokens)
        generated_text = ' '.join([token for token in generated_tokens if token not in ['<START>', '<END>']])
        return generated_text
    
    def calculate_perplexity(self, test_texts: List[str]) -> float:
        """
        Calculate perplexity of the model on test texts.
        
        Args:
            test_texts: List of test text documents
            
        Returns:
            Perplexity score
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before calculating perplexity!")
        
        total_log_prob = 0
        total_words = 0
        
        for text in test_texts:
            tokens = self.preprocess_text(text)
            
            for i in range(1, len(tokens)):  # Skip first token (<START>)
                target_word = tokens[i]
                context_start = max(0, i - self.window_size // 2)
                context_end = i
                context_words = tokens[context_start:context_end]
                
                prob = self.get_word_probability(target_word, context_words)
                if prob > 0:
                    total_log_prob += math.log(prob)
                    total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def get_model_stats(self) -> Dict:
        """
        Get statistics about the trained model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        total_cooccurrences = sum(
            sum(targets.values()) for targets in self.cooccurrence_matrix.values()
        )
        
        return {
            "vocabulary_size": len(self.vocab),
            "total_word_occurrences": sum(self.word_counts.values()),
            "total_cooccurrences": total_cooccurrences,
            "window_size": self.window_size,
            "smoothing_parameter": self.smoothing,
            "unique_context_words": len(self.context_counts)
        }


# Example usage and testing
def demo_window_based_lm():
    """
    Demonstrate the window-based language model with sample data.
    """
    print("=" * 60)
    print("Window-based Language Model Demo")
    print("=" * 60)
    
    # Sample training texts
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a powerful programming language for data science.",
        "Machine learning models can predict future outcomes.",
        "Natural language processing helps computers understand text.",
        "Deep learning uses neural networks to learn patterns.",
        "Data science involves analyzing large datasets.",
        "The fox runs quickly through the forest.",
        "Python programming is fun and easy to learn.",
        "Machine learning algorithms improve with more data.",
        "Language models can generate coherent text.",
        "Neural networks are inspired by the human brain.",
        "Data analysis reveals hidden patterns in information.",
        "The brown fox hunts in the dark forest.",
        "Programming languages help us communicate with computers.",
        "Learning algorithms adapt to new information over time."
    ]
    
    # Initialize and train the model
    lm = WindowBasedLanguageModel(window_size=5, smoothing=1.0)
    lm.train(training_texts)
    
    # Display model statistics
    print("\nModel Statistics:")
    stats = lm.get_model_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test next word prediction
    print("\n" + "="*40)
    print("Next Word Prediction Examples:")
    print("="*40)
    
    test_contexts = [
        "The quick brown",
        "Python is a",
        "Machine learning",
        "Neural networks",
        "Data science"
    ]
    
    for context in test_contexts:
        predictions = lm.predict_next_words(context, top_k=3)
        print(f"\nContext: '{context}'")
        print("Top predictions:")
        for word, prob in predictions:
            print(f"  {word}: {prob:.4f}")
    
    # Test text generation
    print("\n" + "="*40)
    print("Text Generation Examples:")
    print("="*40)
    
    generation_seeds = [
        "The quick brown",
        "Python is",
        "Machine learning"
    ]
    
    for seed in generation_seeds:
        generated = lm.generate_text(seed, max_length=10)
        print(f"\nSeed: '{seed}'")
        print(f"Generated: '{generated}'")
    
    # Calculate perplexity on test data
    test_texts = [
        "The brown fox jumps over the fence.",
        "Python programming is very useful.",
        "Machine learning requires good data."
    ]
    
    perplexity = lm.calculate_perplexity(test_texts)
    print(f"\nPerplexity on test data: {perplexity:.2f}")
    
    return lm


if __name__ == "__main__":
    # Run the demonstration
    model = demo_window_based_lm()
    
    print("\n" + "="*60)
    print("Interactive Mode")
    print("="*60)
    print("Enter text to get next word predictions (or 'quit' to exit):")
    
    while True:
        user_input = input("\nContext: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            try:
                predictions = model.predict_next_words(user_input, top_k=5)
                print("Predictions:")
                for word, prob in predictions:
                    print(f"  {word}: {prob:.4f}")
            except Exception as e:
                print(f"Error: {e}")