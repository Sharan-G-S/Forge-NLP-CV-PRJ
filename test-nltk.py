import nltk

# Download resources (only first time)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# ------------------------------
# Rule-Based POS Tagger
# ------------------------------
def rule_based_pos_tagger(sentence):
    words = nltk.word_tokenize(sentence)
    tags = []
    
    determiners = {"the", "a", "an", "this", "that"}
    pronouns = {"I", "you", "he", "she", "it", "we", "they"}
    prepositions = {"in", "on", "at", "with", "by", "to"}
    
    for word in words:
        lw = word.lower()
        
        if lw in determiners:
            tags.append((word, "DET"))
        elif lw in pronouns:
            tags.append((word, "PRON"))
        elif lw in prepositions:
            tags.append((word, "PREP"))
        elif lw.endswith("ing"):
            tags.append((word, "VBG"))
        elif lw.endswith("ed"):
            tags.append((word, "VBD"))
        elif lw.endswith("ly"):
            tags.append((word, "ADV"))
        elif lw.endswith("s"):
            tags.append((word, "NNS"))  # plural noun
        else:
            tags.append((word, "NOUN"))  # default
    return tags

# ------------------------------
# Statistical POS Tagger (NLTK)
# ------------------------------
def statistical_pos_tagger(sentence):
    words = nltk.word_tokenize(sentence)
    return nltk.pos_tag(words)

# ------------------------------
# Test Sentences
# ------------------------------
sentences = [
    "The cat is chasing the mouse",
    "I saw the man with the telescope",
    "She runs fast"
]

# ------------------------------
# Run Both Taggers
# ------------------------------
for s in sentences:
    print("\nSentence:", s)
    print("Rule-based:", rule_based_pos_tagger(s))
    print("Statistical:", statistical_pos_tagger(s))
