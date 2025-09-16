#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re, nltk, numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def preprocess_en(text: str):
    tokens = nltk.word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    return set(lemmas)

def synonyms_en(word: str):
    syns = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().lower())
    return syns

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def semantic_hit(keyword: str, generated: str, sbert: SentenceTransformer, base_thresh=0.60):
    kw = normalize(keyword); gt = normalize(generated)
    if not kw or not gt: return False
    if len(kw) <= 2: return False
    v1 = sbert.encode([kw], convert_to_numpy=True, normalize_embeddings=True)[0]
    v2 = sbert.encode([gt], convert_to_numpy=True, normalize_embeddings=True)[0]
    return cosine(v1, v2) >= base_thresh

def covered_ratio(keywords, generated: str, sbert: SentenceTransformer) -> float:
    gen_norm = normalize(generated)
    if not keywords: return 0.0
    covered = 0
    gen_lemmas = preprocess_en(gen_norm)
    for raw_kw in keywords:
        kw_norm = normalize(raw_kw)
        if not kw_norm: continue
        lex_hit = (kw_norm in gen_norm)
        syn_hit = False
        kw_lemmas = preprocess_en(kw_norm)
        syns = set()
        for w in kw_lemmas:
            syns |= synonyms_en(w); syns.add(w)
        if syns:
            syn_lemmas = set()
            for s in syns:
                s = s.replace("_", " ")
                for tok in nltk.word_tokenize(s):
                    syn_lemmas.add(lemmatizer.lemmatize(tok))
            syn_hit = len(gen_lemmas & syn_lemmas) > 0
        sem_hit = semantic_hit(kw_norm, gen_norm, sbert, base_thresh=0.60)
        if lex_hit or syn_hit or sem_hit:
            covered += 1
    return covered / max(1, len(keywords))

