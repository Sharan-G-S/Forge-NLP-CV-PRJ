"""
Statistical Machine Translation (SMT) System
Implements a complete SMT pipeline with word alignment, phrase extraction, and translation
"""

import numpy as np
from collections import defaultdict, Counter
import re
import math
from typing import List, Dict, Tuple, Optional, Set
import itertools

class WordAlignmentModel:
    """
    IBM Model 1 for word alignment - foundational model for SMT
    """
    
    def __init__(self, num_iterations: int = 10):
        self.num_iterations = num_iterations
        self.translation_probs = defaultdict(lambda: defaultdict(float))
        self.source_vocab = set()
        self.target_vocab = set()
        
    def tokenize(self, sentence: str) -> List[str]:
        """Tokenize sentence into words with Tamil script support"""
        # Basic tokenization that works for both English and Tamil
        # Tamil words are separated by spaces in our data
        words = sentence.strip().split()
        # Convert to lowercase for English, keep Tamil as-is
        processed_words = []
        for word in words:
            # Check if word contains Tamil characters (Unicode range U+0B80-U+0BFF)
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                processed_words.append(word)  # Keep Tamil words as-is
            else:
                processed_words.append(word.lower())  # Lowercase English words
        return processed_words
    
    def initialize_uniform_probabilities(self, parallel_corpus: List[Tuple[str, str]]):
        """Initialize translation probabilities uniformly"""
        # Collect vocabularies
        for source_sent, target_sent in parallel_corpus:
            source_words = self.tokenize(source_sent)
            target_words = self.tokenize(target_sent)
            
            self.source_vocab.update(source_words)
            self.target_vocab.update(target_words)
        
        # Initialize uniform probabilities
        uniform_prob = 1.0 / len(self.target_vocab)
        for s_word in self.source_vocab:
            for t_word in self.target_vocab:
                self.translation_probs[s_word][t_word] = uniform_prob
    
    def train(self, parallel_corpus: List[Tuple[str, str]]):
        """Train IBM Model 1 using EM algorithm"""
        print(f"Training word alignment model with {len(parallel_corpus)} sentence pairs...")
        
        # Initialize probabilities
        self.initialize_uniform_probabilities(parallel_corpus)
        
        for iteration in range(self.num_iterations):
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            
            # E-step: compute expected counts
            count = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)
            
            for source_sent, target_sent in parallel_corpus:
                source_words = self.tokenize(source_sent)
                target_words = self.tokenize(target_sent)
                
                for s_word in source_words:
                    # Compute normalization constant
                    s_total = sum(self.translation_probs[s_word][t_word] 
                                for t_word in target_words)
                    
                    if s_total > 0:
                        for t_word in target_words:
                            # Fractional count
                            c = self.translation_probs[s_word][t_word] / s_total
                            count[s_word][t_word] += c
                            total[s_word] += c
            
            # M-step: update probabilities
            for s_word in self.source_vocab:
                if total[s_word] > 0:
                    for t_word in self.target_vocab:
                        self.translation_probs[s_word][t_word] = \
                            count[s_word][t_word] / total[s_word]
        
        print("Word alignment training completed!")
    
    def get_best_alignment(self, source_sent: str, target_sent: str) -> List[Tuple[int, int]]:
        """Get best word alignment for sentence pair"""
        source_words = self.tokenize(source_sent)
        target_words = self.tokenize(target_sent)
        
        alignments = []
        for i, s_word in enumerate(source_words):
            best_j = -1
            best_prob = 0
            
            for j, t_word in enumerate(target_words):
                prob = self.translation_probs[s_word][t_word]
                if prob > best_prob:
                    best_prob = prob
                    best_j = j
            
            if best_j != -1:
                alignments.append((i, best_j))
        
        return alignments

class PhraseExtractor:
    """
    Extract phrase pairs from word-aligned parallel corpus
    """
    
    def __init__(self, max_phrase_length: int = 7):
        self.max_phrase_length = max_phrase_length
        self.phrase_pairs = []
        
    def extract_phrases(self, source_sent: str, target_sent: str, 
                       alignments: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
        """Extract consistent phrase pairs from aligned sentence pair"""
        # Handle mixed script tokenization
        source_words = []
        target_words = []
        
        for word in source_sent.strip().split():
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                source_words.append(word)
            else:
                source_words.append(word.lower())
                
        for word in target_sent.strip().split():
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                target_words.append(word)
            else:
                target_words.append(word.lower())
        
        # Convert alignments to sets for faster lookup
        alignment_set = set(alignments)
        
        phrase_pairs = []
        
        # Try all possible source phrases
        for start_s in range(len(source_words)):
            for end_s in range(start_s + 1, min(len(source_words) + 1, 
                                               start_s + self.max_phrase_length + 1)):
                
                # Find aligned target positions
                target_positions = set()
                for i in range(start_s, end_s):
                    for j in range(len(target_words)):
                        if (i, j) in alignment_set:
                            target_positions.add(j)
                
                if not target_positions:
                    continue
                
                # Find consistent target phrase boundaries
                start_t = min(target_positions)
                end_t = max(target_positions) + 1
                
                # Check consistency: no alignments outside the target phrase
                consistent = True
                for i in range(len(source_words)):
                    if i < start_s or i >= end_s:  # Outside source phrase
                        for j in range(start_t, end_t):  # Inside target phrase
                            if (i, j) in alignment_set:
                                consistent = False
                                break
                    if not consistent:
                        break
                
                if consistent and end_t - start_t <= self.max_phrase_length:
                    source_phrase = ' '.join(source_words[start_s:end_s])
                    target_phrase = ' '.join(target_words[start_t:end_t])
                    phrase_pairs.append((source_phrase, target_phrase))
        
        return phrase_pairs
    
    def build_phrase_table(self, parallel_corpus: List[Tuple[str, str]], 
                          word_aligner: WordAlignmentModel) -> Dict[str, Dict[str, float]]:
        """Build phrase translation table with probabilities"""
        print("Extracting phrase pairs...")
        
        # Extract all phrase pairs
        all_phrase_pairs = []
        for i, (source_sent, target_sent) in enumerate(parallel_corpus):
            if i % 100 == 0:
                print(f"Processing sentence pair {i+1}/{len(parallel_corpus)}")
            
            alignments = word_aligner.get_best_alignment(source_sent, target_sent)
            phrase_pairs = self.extract_phrases(source_sent, target_sent, alignments)
            all_phrase_pairs.extend(phrase_pairs)
        
        # Count phrase pair frequencies
        phrase_counts = Counter(all_phrase_pairs)
        source_phrase_counts = Counter([source for source, _ in all_phrase_pairs])
        
        # Calculate translation probabilities
        phrase_table = defaultdict(lambda: defaultdict(float))
        for (source_phrase, target_phrase), count in phrase_counts.items():
            phrase_table[source_phrase][target_phrase] = \
                count / source_phrase_counts[source_phrase]
        
        print(f"Extracted {len(phrase_counts)} unique phrase pairs")
        return dict(phrase_table)

class LanguageModel:
    """
    N-gram language model for target language fluency
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab = set()
    
    def tokenize(self, sentence: str) -> List[str]:
        """Tokenize and add boundary markers with Tamil support"""
        words = []
        for word in sentence.strip().split():
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                words.append(word)  # Keep Tamil words as-is
            else:
                words.append(word.lower())  # Lowercase English words
        return ['<s>'] * (self.n - 1) + words + ['</s>']
    
    def train(self, sentences: List[str]):
        """Train n-gram language model"""
        print(f"Training {self.n}-gram language model...")
        
        for sentence in sentences:
            words = self.tokenize(sentence)
            self.vocab.update(words)
            
            for i in range(len(words) - self.n + 1):
                ngram = tuple(words[i:i + self.n])
                context = tuple(words[i:i + self.n - 1])
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        
        print(f"Language model trained on {len(self.ngram_counts)} n-grams")
    
    def get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """Get probability of word given context with smoothing"""
        ngram = context + (word,)
        
        # Add-1 smoothing
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]
        
        return (ngram_count + 1) / (context_count + len(self.vocab))
    
    def score_sentence(self, sentence: str) -> float:
        """Score sentence using language model"""
        words = self.tokenize(sentence)
        log_prob = 0
        
        for i in range(self.n - 1, len(words)):
            context = tuple(words[i - self.n + 1:i])
            word = words[i]
            prob = self.get_probability(word, context)
            log_prob += math.log(prob)
        
        return log_prob

class StatisticalMachineTranslator:
    """
    Complete Statistical Machine Translation system
    """
    
    def __init__(self, max_phrase_length: int = 7, lm_order: int = 3):
        self.word_aligner = WordAlignmentModel()
        self.phrase_extractor = PhraseExtractor(max_phrase_length)
        self.language_model = LanguageModel(lm_order)
        self.phrase_table = {}
        self.is_trained = False
    
    def train(self, parallel_corpus: List[Tuple[str, str]], 
              target_monolingual: List[str]):
        """Train all components of the SMT system"""
        print("="*60)
        print("Training Statistical Machine Translation System")
        print("="*60)
        
        # Train word alignment model
        self.word_aligner.train(parallel_corpus)
        
        # Extract phrase table
        self.phrase_table = self.phrase_extractor.build_phrase_table(
            parallel_corpus, self.word_aligner)
        
        # Train language model on target side + monolingual data
        target_sentences = [target for _, target in parallel_corpus]
        all_target_sentences = target_sentences + target_monolingual
        self.language_model.train(all_target_sentences)
        
        self.is_trained = True
        print("SMT system training completed!")
    
    def generate_translation_options(self, source_sentence: str) -> List[List[Tuple[str, float]]]:
        """Generate all possible phrase translations for source sentence"""
        # Handle mixed script tokenization
        source_words = []
        for word in source_sentence.strip().split():
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                source_words.append(word)
            else:
                source_words.append(word.lower())
        
        # For each position, find all possible phrase translations
        translation_options = [[] for _ in range(len(source_words))]
        
        for start in range(len(source_words)):
            for end in range(start + 1, min(len(source_words) + 1, 
                                          start + self.phrase_extractor.max_phrase_length + 1)):
                source_phrase = ' '.join(source_words[start:end])
                
                if source_phrase in self.phrase_table:
                    for target_phrase, prob in self.phrase_table[source_phrase].items():
                        translation_options[start].append((target_phrase, prob))
        
        return translation_options
    
    def beam_search_translate(self, source_sentence: str, beam_size: int = 5) -> List[Tuple[str, float]]:
        """Translate using beam search with phrase-based model"""
        if not self.is_trained:
            raise ValueError("SMT system must be trained before translation!")
        
        # Handle mixed script tokenization  
        source_words = []
        for word in source_sentence.strip().split():
            if any('\u0b80' <= char <= '\u0bff' for char in word):
                source_words.append(word)
            else:
                source_words.append(word.lower())
        
        # Simple greedy translation for demonstration
        # In practice, you'd implement proper beam search with coverage
        translation_parts = []
        
        i = 0
        while i < len(source_words):
            best_translation = None
            best_score = float('-inf')
            best_length = 1
            
            # Try phrases of different lengths starting at position i
            for length in range(1, min(self.phrase_extractor.max_phrase_length + 1, 
                                     len(source_words) - i + 1)):
                source_phrase = ' '.join(source_words[i:i + length])
                
                if source_phrase in self.phrase_table:
                    for target_phrase, translation_prob in self.phrase_table[source_phrase].items():
                        # Simple scoring: translation probability only
                        # In practice, you'd combine with language model score
                        score = math.log(translation_prob)
                        
                        if score > best_score:
                            best_score = score
                            best_translation = target_phrase
                            best_length = length
            
            if best_translation:
                translation_parts.append(best_translation)
                i += best_length
            else:
                # Fallback: word-by-word translation
                source_word = source_words[i]
                if source_word in self.word_aligner.translation_probs:
                    best_target_word = max(
                        self.word_aligner.translation_probs[source_word].items(),
                        key=lambda x: x[1]
                    )[0]
                    translation_parts.append(best_target_word)
                else:
                    translation_parts.append(source_word)  # Copy unknown words
                i += 1
        
        translation = ' '.join(translation_parts)
        
        # Score with language model
        lm_score = self.language_model.score_sentence(translation)
        
        return [(translation, lm_score)]
    
    def translate(self, source_sentence: str) -> str:
        """Translate a source sentence to target language"""
        translations = self.beam_search_translate(source_sentence, beam_size=1)
        return translations[0][0] if translations else source_sentence
    
    def get_system_stats(self) -> Dict:
        """Get statistics about the trained SMT system"""
        if not self.is_trained:
            return {"status": "System not trained"}
        
        return {
            "source_vocabulary_size": len(self.word_aligner.source_vocab),
            "target_vocabulary_size": len(self.word_aligner.target_vocab),
            "phrase_table_size": sum(len(targets) for targets in self.phrase_table.values()),
            "unique_source_phrases": len(self.phrase_table),
            "language_model_order": self.language_model.n,
            "language_model_ngrams": len(self.language_model.ngram_counts)
        }

# Demo function
def demo_smt_system():
    """Demonstrate the Statistical Machine Translation system"""
    print("="*70)
    print("Statistical Machine Translation System Demo (English-Tamil)")
    print("="*70)
    
    # Sample parallel corpus (English-Tamil for demonstration)
    # Note: Tamil uses different script and grammar structure
    parallel_corpus = [
        ("the cat is on the mat", "பூனை பாய் மீது உள்ளது"),
        ("the dog runs fast", "நாய் வேகமாக ஓடுகிறது"),
        ("I love programming", "எனக்கு நிரலாக்கம் பிடிக்கும்"),
        ("the book is good", "புத்தகம் நல்லது"),
        ("she speaks tamil", "அவள் தமிழ் பேசுகிறாள்"),
        ("the car is red", "கார் சிவப்பு நிறம்"),
        ("he works every day", "அவன் தினமும் வேலை செய்கிறான்"),
        ("the house is big", "வீடு பெரியது"),
        ("we eat rice together", "நாம் ஒன்றாக சாதம் சாப்பிடுகிறோம்"),
        ("the weather is nice", "வானிலை நன்றாக உள்ளது"),
        ("I study computer science", "நான் கணினி அறிவியல் படிக்கிறேன்"),
        ("the cat sleeps peacefully", "பூனை அமைதியாக தூங்குகிறது"),
        ("she reads many books", "அவள் பல புத்தகங்கள் படிக்கிறாள்"),
        ("the dog plays in the park", "நாய் பூங்காவில் விளையாடுகிறது"),
        ("he drives a blue car", "அவன் நீல நிற கார் ஓட்டுகிறான்"),
        ("water is essential", "தண்ணீர் அவசியம்"),
        ("the sun rises in the east", "சூரியன் கிழக்கில் உதிக்கிறது"),
        ("children play outside", "குழந்தைகள் வெளியில் விளையாடுகிறார்கள்"),
        ("food tastes delicious", "உணவு சுவையாக உள்ளது"),
        ("my name is john", "என் பெயர் ஜான்")
    ]
    
    # Monolingual target data for language model (Tamil)
    target_monolingual = [
        "பூனை படுக்கையில் தூங்குகிறது",
        "நாய் தோட்டத்தில் ஓடுகிறது", 
        "வீட்டில் பல ஜன்னல்கள் உள்ளன",
        "அவள் பல்கலைக்கழகத்தில் படிக்கிறாள்",
        "அவன் அலுவலகத்தில் வேலை செய்கிறான்",
        "உணவு மிகவும் சுவையாக உள்ளது",
        "இன்று வானிலை அருமையாக உள்ளது",
        "புத்தகங்கள் மேசையில் உள்ளன",
        "இசை நன்றாக ஒலிக்கிறது",
        "காருக்கு பெட்ரோல் தேவை",
        "மாணவர்கள் வகுப்பில் கவனம் செலுத்துகிறார்கள்",
        "ஆசிரியர் பாடம் நடத்துகிறார்",
        "மழை பெய்கிறது",
        "பறவைகள் வானில் பறக்கின்றன",
        "மலர்கள் அழகாக பூக்கின்றன"
    ]
    
    # Initialize and train SMT system
    smt = StatisticalMachineTranslator(max_phrase_length=6, lm_order=3)
    smt.train(parallel_corpus, target_monolingual)
    
    # Display system statistics
    print("\nSMT System Statistics (English-Tamil):")
    stats = smt.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test translations
    print("\n" + "="*50)
    print("Translation Examples (English -> Tamil):")
    print("="*50)
    
    test_sentences = [
        "the cat is good",
        "the dog runs", 
        "I love books",
        "the house is big",
        "she speaks tamil",
        "water is essential",
        "the sun rises",
        "children play outside"
    ]
    
    for source in test_sentences:
        translation = smt.translate(source)
        print(f"English:  '{source}'")
        print(f"Tamil:    '{translation}'")
        print("-" * 50)
    
    # Show some phrase table entries
    print("\nSample Phrase Table Entries (English -> Tamil):")
    print("-" * 50)
    phrase_count = 0
    for source_phrase, translations in smt.phrase_table.items():
        if phrase_count >= 15:
            break
        best_translation = max(translations.items(), key=lambda x: x[1])
        print(f"'{source_phrase}' -> '{best_translation[0]}' (p={best_translation[1]:.3f})")
        phrase_count += 1
    
    return smt

if __name__ == "__main__":
    # Run the demonstration
    smt_system = demo_smt_system()
    
    print("\n" + "="*70)
    print("Interactive Translation Mode (English -> Tamil)")
    print("="*70)
    print("Enter English sentences to translate to Tamil (or 'quit' to exit):")
    print("Examples: 'the cat sleeps', 'I love programming', 'the weather is nice'")
    
    while True:
        user_input = input("\nEnglish: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if user_input:
            try:
                translation = smt_system.translate(user_input)
                print(f"Tamil:   {translation}")
            except Exception as e:
                print(f"Translation error: {e}")