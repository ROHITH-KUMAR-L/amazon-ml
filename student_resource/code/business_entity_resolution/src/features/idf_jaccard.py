"""
IDF-Weighted Token Jaccard Module for Business Entity Resolution.
Weights token overlap by Inverse Document Frequency (IDF) so rare brand names carry higher weight.
"""

from collections import Counter
import math
from typing import List, Set
import pandas as pd


class IDFEstimator:
    def __init__(self):
        self.idf_dict = {}
        self.default_idf = 1.0

    def fit(self, text_series: pd.Series):
        doc_count = len(text_series)
        df_counter = Counter()

        for text in text_series:
            unique_tokens = set(str(text).split())
            for token in unique_tokens:
                df_counter[token] += 1

        for token, count in df_counter.items():
            # Smooth IDF: log((N + 1) / (df + 1)) + 1
            self.idf_dict[token] = math.log((doc_count + 1.0) / (count + 1.0)) + 1.0

        self.default_idf = math.log(doc_count + 1.0) + 1.0

    def get_idf(self, token: str) -> float:
        return self.idf_dict.get(token, self.default_idf)

    def weighted_jaccard(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0

        t1 = set(s1.split())
        t2 = set(s2.split())

        inter = t1.intersection(t2)
        union = t1.union(t2)

        if not union:
            return 0.0

        inter_weight = sum(self.get_idf(t) for t in inter)
        union_weight = sum(self.get_idf(t) for t in union)

        if union_weight == 0:
            return 0.0

        return inter_weight / union_weight
