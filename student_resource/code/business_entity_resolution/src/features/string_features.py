"""
String Similarity Features Module for Business Entity Resolution.
Computes string metrics (Levenshtein ratio, Token-Sort ratio, Token-Set ratio, Char N-Gram similarity).
"""

from difflib import SequenceMatcher
import numpy as np


def levenshtein_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    return SequenceMatcher(None, s1, s2).ratio()


def token_sort_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return SequenceMatcher(None, t1, t2).ratio()


def token_set_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    
    intersection = tokens1.intersection(tokens2)
    if not intersection:
        return 0.0
        
    sorted_inter = " ".join(sorted(intersection))
    sorted_t1 = " ".join(sorted(tokens1))
    sorted_t2 = " ".join(sorted(tokens2))
    
    r1 = SequenceMatcher(None, sorted_inter, sorted_t1).ratio()
    r2 = SequenceMatcher(None, sorted_inter, sorted_t2).ratio()
    r3 = SequenceMatcher(None, sorted_t1, sorted_t2).ratio()
    
    return max(r1, r2, r3)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    if len(s1) < n or len(s2) < n:
        return 0.0
    ngrams1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    ngrams2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    
    union_len = len(ngrams1.union(ngrams2))
    if union_len == 0:
        return 0.0
    return len(ngrams1.intersection(ngrams2)) / union_len
