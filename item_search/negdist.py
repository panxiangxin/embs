from __future__ import annotations

import bisect
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NegDistribution:
    samples: list[float]

    def cdf(self, x: float) -> float:
        x = float(max(-1.0, min(1.0, x)))
        if not self.samples:
            return 1.0 if x >= 1.0 else 0.0
        idx = bisect.bisect_right(self.samples, x)
        return idx / len(self.samples)

    def tail_p_max(self, x: float, m: int) -> float:
        """
        p_spurious = P(max >= x) under null, assuming i.i.d. draws from this distribution:
          p = 1 - CDF(x)^m
        """
        # Numeric guard: for unit-normalized float32 embeddings, an exact/self match can land slightly below 1.0.
        # If we keep x as-is, and the sampled negative distribution contains any 1.0 values (e.g., duplicate labels),
        # CDF(x) < 1 and p can become spuriously large after the multiple-comparisons correction.
        x = float(max(-1.0, min(1.0, x)))
        if x >= 1.0 - 1e-6:
            return 0.0
        if m <= 1:
            return 1.0 - self.cdf(x)
        c = self.cdf(x)
        c = max(0.0, min(1.0, c))
        return 1.0 - (c**m)


def build_neg_distribution(vectors: np.ndarray, max_pairs: int = 50_000, seed: int = 7) -> NegDistribution:
    """
    Build an empirical "negative" cosine similarity distribution by sampling random
    pairs among existing normalized vectors.
    """
    if vectors is None:
        return NegDistribution(samples=[])
    vecs = np.asarray(vectors, dtype=np.float32)
    if vecs.ndim != 2 or vecs.shape[0] < 2:
        return NegDistribution(samples=[])

    n = vecs.shape[0]
    total_pairs = (n * (n - 1)) // 2
    draws = int(min(max_pairs, total_pairs))
    draws = max(200, draws) if total_pairs >= 200 else int(total_pairs)
    if draws <= 0:
        return NegDistribution(samples=[])

    rng = random.Random(seed)
    sims: list[float] = []
    for _ in range(draws):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        sim = float(np.dot(vecs[i], vecs[j]))
        # Numeric guard: cosine similarity may slightly exceed [-1, 1] due to float error.
        sims.append(float(max(-1.0, min(1.0, sim))))

    sims.sort()
    return NegDistribution(samples=sims)
