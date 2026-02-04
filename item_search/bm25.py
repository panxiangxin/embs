from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BM25Index:
    docs: list[list[str]]
    doc_freq: dict[str, int]
    doc_len: list[int]
    avgdl: float
    idf: dict[str, float]
    k1: float = 1.2
    b: float = 0.75

    @staticmethod
    def build(docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> "BM25Index":
        doc_freq: dict[str, int] = {}
        doc_len = [len(d) for d in docs]
        N = len(docs) or 1
        for doc in docs:
            for term in set(doc):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        idf: dict[str, float] = {}
        for term, df in doc_freq.items():
            idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        avgdl = sum(doc_len) / N
        return BM25Index(docs=docs, doc_freq=doc_freq, doc_len=doc_len, avgdl=avgdl, idf=idf, k1=k1, b=b)

    def score_all(self, query_tokens: list[str]) -> list[float]:
        if not query_tokens:
            return [0.0] * len(self.docs)
        q = [t for t in query_tokens if t in self.idf]
        if not q:
            return [0.0] * len(self.docs)

        scores = [0.0] * len(self.docs)
        q_terms = Counter(q)

        for doc_idx, doc in enumerate(self.docs):
            freqs = Counter(doc)
            dl = self.doc_len[doc_idx] or 1
            denom_const = self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            s = 0.0
            for term, qtf in q_terms.items():
                f = freqs.get(term, 0)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                s += (idf * qtf) * (f * (self.k1 + 1.0)) / (f + denom_const)
            scores[doc_idx] = s
        return scores

    @staticmethod
    def topn(scores: list[float], n: int) -> list[tuple[int, float]]:
        if n <= 0:
            return []
        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:n]

