"""
Simple Python examples for Word-to-Vector conversion
"""

# Method 1: Using scikit-learn's TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tfidf_word_vectors():
    """Convert words to vectors using TF-IDF"""
    # Sample documents
    documents = [
        "I love machine learning",
        "Python is great for data science",
        "Machine learning with Python is fun",
        "Data science and machine learning"
    ]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()
    
    print("TF-IDF Word Vectors:")
    print("Feature names:", feature_names)
    print("TF-IDF Matrix shape:", tfidf_matrix.shape)
    print("TF-IDF Matrix (dense):")
    print(tfidf_matrix.toarray())
    print("-" * 50)

# Method 2: Using scikit-learn's CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def count_word_vectors():
    """Convert words to vectors using Count Vectorization"""
    documents = [
        "I love machine learning",
        "Python is great for data science", 
        "Machine learning with Python is fun"
    ]
    
    # Create count vectorizer
    vectorizer = CountVectorizer()
    
    # Fit and transform documents
    count_matrix = vectorizer.fit_transform(documents)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    print("Count Vectorization:")
    print("Feature names:", feature_names)
    print("Count Matrix:")
    print(count_matrix.toarray())
    print("-" * 50)

# Method 3: Simple one-hot encoding approach
def simple_one_hot_encoding():
    """Simple one-hot encoding for words"""
    sentences = ["I love Python", "Python is great", "I love programming"]
    
    # Create vocabulary
    vocabulary = set()
    for sentence in sentences:
        vocabulary.update(sentence.lower().split())
    
    vocabulary = sorted(list(vocabulary))
    word_to_index = {word: i for i, word in enumerate(vocabulary)}
    
    print("Simple One-Hot Encoding:")
    print("Vocabulary:", vocabulary)
    print("Word to Index mapping:", word_to_index)
    
    # Convert sentences to vectors
    for sentence in sentences:
        vector = [0] * len(vocabulary)
        words = sentence.lower().split()
        for word in words:
            if word in word_to_index:
                vector[word_to_index[word]] = 1
        print(f"'{sentence}' -> {vector}")
    
    print("-" * 50)

# Method 4: Using Word2Vec with gensim (if installed)
def word2vec_example():
    """Word2Vec example using gensim"""
    try:
        from gensim.models import Word2Vec
        
        # Sample sentences (list of lists of words)
        sentences = [
            ["I", "love", "machine", "learning"],
            ["Python", "is", "great", "for", "data", "science"],
            ["Machine", "learning", "with", "Python", "is", "fun"],
            ["Data", "science", "and", "machine", "learning"]
        ]
        
        # Train Word2Vec model
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        print("Word2Vec Vectors:")
        print("Vocabulary size:", len(model.wv.key_to_index))
        print("Vector for 'machine':")
        print(model.wv['machine'][:10])  # Show first 10 dimensions
        
        # Find similar words
        try:
            similar_words = model.wv.most_similar('machine', topn=3)
            print("Words similar to 'machine':", similar_words)
        except:
            print("Not enough data for similarity calculation")
            
    except ImportError:
        print("Gensim not installed. Install with: pip install gensim")
    
    print("-" * 50)

if __name__ == "__main__":
    print("Word-to-Vector Examples\n")
    
    # Run all examples
    tfidf_word_vectors()
    count_word_vectors()
    simple_one_hot_encoding()
    word2vec_example()
    
    print("\nTo install required packages:")
    print("pip install scikit-learn gensim numpy")