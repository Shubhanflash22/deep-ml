
import numpy as np
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:

    if not candidate or not references or max_n <= 0:
        return 0.0

    cor = 0
    incor = 0
    n = len(candidate)

    # Step 1: Unigram precision
    for i in range(n):
        if candidate[i] in references[0]:
            cor += 1
        else:
            incor += 1

    unigram_precision = cor / (cor + incor)

    # Step 2: Bigram precision
    cor = 0
    incor = 0

    reference_bigrams = [
        (ref[j], ref[j + 1])
        for ref in references
        for j in range(len(ref) - 1)
    ]

    for i in range(n - 1):
        candidate_bigram = (candidate[i], candidate[i + 1])

        if candidate_bigram in reference_bigrams:
            cor += 1
        else:
            incor += 1

    bigram_precision = cor / (cor + incor) if cor + incor > 0 else 0.0

    # Step 3: Calculate modified n-gram precision
    precisions = []

    for order in range(1, max_n + 1):

        candidate_ngrams = Counter(
            tuple(candidate[i:i + order])
            for i in range(n - order + 1)
        )

        # Get maximum n-gram counts across all references
        max_ref_counts = Counter()

        for ref in references:
            ref_ngrams = Counter(
                tuple(ref[i:i + order])
                for i in range(len(ref) - order + 1)
            )

            for gram, count in ref_ngrams.items():
                max_ref_counts[gram] = max(
                    max_ref_counts[gram], count
                )

        # Count clipped matches
        cor = sum(
            min(count, max_ref_counts[gram])
            for gram, count in candidate_ngrams.items()
        )

        total = sum(candidate_ngrams.values())

        if total == 0 or cor == 0:
            return 0.0

        precision = cor / total
        precisions.append(precision)

    # Step 4: Geometric mean
    log_sum = sum(np.log(p) for p in precisions)
    geo_mean = np.exp(log_sum / max_n)

    # Step 5: Brevity penalty
    c = len(candidate)

    r = min(
        (len(ref) for ref in references),
        key=lambda length: (abs(length - c), length)
    )

    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    # Step 6: Final BLEU score
    bleu = bp * geo_mean
    return float(bleu)