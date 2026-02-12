# syntactic_parser_semantic_similarity.py
import re
from pprint import pprint
from nltk import CFG, ChartParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- robust tokenizer (tries nltk.word_tokenize; falls back otherwise) ----
def safe_tokenize(text):
    try:
        from nltk import word_tokenize as _wt
        toks = _wt(text)
        return toks
    except Exception:
        # simple fallback: words and punctuation tokens
        toks = re.findall(r"[\w']+|[.,!?;]", text)
        return toks

# ---- grammar and parser (small CFG for simple sentences) ----
grammar_text = """
S -> NP VP
NP -> Det N | Det N PP | 'I' | 'you' | N
VP -> V NP | V NP PP | V PP | V
PP -> P NP
Det -> 'the' | 'a' | 'an' | 'my' | 'your'
N -> 'cat' | 'dog' | 'park' | 'telescope' | 'man' | 'woman' | 'apple' | 'boy' | 'girl' | 'car' | 'pear'
V -> 'saw' | 'ate' | 'walked' | 'likes' | 'loves' | 'drives' | 'drove'
P -> 'with' | 'in' | 'on' | 'by' | 'to'
"""
grammar = CFG.fromstring(grammar_text)
parser = ChartParser(grammar)

def parse_sentence(sentence):
    toks = safe_tokenize(sentence)
    toks = [t for t in toks if re.search(r'\w', t)]  # remove pure punctuation tokens
    toks_for_parse = [t if t in ("I", "you") else t.lower() for t in toks]
    try:
        trees = list(parser.parse(toks_for_parse))
    except Exception:
        trees = []
    return toks_for_parse, trees

# ---- semantic similarity: TF-IDF cosine (reliable, no heavy deps) ----
def tfidf_cosine_similarity(sent1, sent2, corpus=None):
    docs = [sent1, sent2] if corpus is None else corpus + [sent1, sent2]
    vect = TfidfVectorizer().fit(docs)
    vecs = vect.transform([sent1, sent2])
    sim = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
    return float(sim)

# ---- optional WordNet-based sentence similarity (if data present) ----
def wordnet_sentence_similarity(sent1, sent2):
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return None  # environment doesn't have WordNet installed/downloaded

    # build content-word list using POS tagger if available, else a heuristic fallback
    def content_words(sentence):
        toks = [t for t in safe_tokenize(sentence) if any(c.isalnum() for c in t)]
        try:
            from nltk import pos_tag as _pt
            tagged = _pt(toks)
        except Exception:
            # fallback heuristic tagging
            tagged = []
            for t in toks:
                if t.lower() in ('the','a','an','my','your','i','you','he','she','it','they','we'):
                    tagged.append((t, 'DT'))
                elif re.match(r'.+ed$', t.lower()):
                    tagged.append((t, 'VBD'))
                elif re.match(r'.+ing$', t.lower()):
                    tagged.append((t, 'VBG'))
                elif re.match(r'.+s$', t.lower()) and len(t) > 3:
                    tagged.append((t, 'NNS'))
                else:
                    tagged.append((t, 'NN'))
        content = []
        for w, pos in tagged:
            pos_low = pos[0].lower()
            if pos_low in ('n', 'v', 'j', 'r'):
                content.append((w.lower(), pos_low))
        return content

    def wn_pos(pos_char):
        return {'n': wn.NOUN, 'v': wn.VERB, 'j': wn.ADJ, 'r': wn.ADV}.get(pos_char, wn.NOUN)

    cw1 = content_words(sent1)
    cw2 = content_words(sent2)
    if not cw1 or not cw2:
        return 0.0

    def max_sim_for(word, pos, targets):
        synsets = wn.synsets(word, pos=wn_pos(pos))
        if not synsets:
            return 0.0
        best = 0.0
        for tword, tpos in targets:
            tsyns = wn.synsets(tword, pos=wn_pos(tpos))
            for s in synsets:
                for t in tsyns:
                    try:
                        val = s.path_similarity(t) or 0.0
                    except Exception:
                        val = 0.0
                    if val > best:
                        best = val
        return best

    sims1 = [max_sim_for(w, p, cw2) for (w, p) in cw1]
    sims2 = [max_sim_for(w, p, cw1) for (w, p) in cw2]
    avg = (sum(sims1) / len(sims1) + sum(sims2) / len(sims2)) / 2.0
    return float(avg)

# ---- small demo ----
if __name__ == "__main__":
    examples = [
        "I saw the man with the telescope",
        "The dog walked in the park",
        "She ate an apple",
        "The boy loves the girl",
        "My car drove to the park"
    ]
    print("=== Parsing demo ===")
    for s in examples:
        toks, trees = parse_sentence(s)
        print("Sentence:", s)
        print("Tokens used for parse:", toks)
        if trees:
            print("Number of parses:", len(trees))
            for t in trees:
                t.pretty_print()
        else:
            print("No parse from this grammar.")
        print("-" * 40)

    pairs = [
        ("I saw the man with the telescope", "I saw the man in the park"),
        ("The dog walked in the park", "A man walked in the park"),
        ("She ate an apple", "He ate a pear"),
        ("The boy loves the girl", "The girl loves the boy"),
        ("My car drove to the park", "I drove my car to the park")
    ]

    print("\n=== TF-IDF cosine similarity demo ===")
    for a, b in pairs:
        print(f"\nA: {a}\nB: {b}")
        print("TF-IDF cosine:", tfidf_cosine_similarity(a, b))

    print("\n=== WordNet-based similarity (optional) ===")
    for a, b in pairs:
        wns = wordnet_sentence_similarity(a, b)
        if wns is None:
            print("WordNet data not available in this environment; skipping WordNet similarity.")
            break
        else:
            print(f"A: {a}\nB: {b}\nWordNet sim: {wns}")
s