import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenized_word(w):
    return nltk.word_tokenize(w)

def stem(w):
    return stemmer.stem(w.lower())

def bag_of_word(s,all_words):
    s = [stem(w) for w in s ]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for i, n in enumerate(all_words):
        if n in s:
            bag[i] = 1.0
    return bag

    
