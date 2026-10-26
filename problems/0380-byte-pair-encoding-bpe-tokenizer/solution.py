def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus.

    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.

    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    merges = []

    # Convert the corpus into token sequences and preserve their frequencies.
    word_freqs = {
        tuple(word.split()): freq
        for word, freq in corpus.items()
    }

    for _ in range(num_merges):
        # Count the frequencies of all adjacent token pairs.
        pair_freqs = {}

        for tokens, freq in word_freqs.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

        # Stop if there are no adjacent pairs left.
        if not pair_freqs:
            break

        # Find the most frequent pair.
        # In case of ties, the first encountered pair is selected.
        best_pair = max(pair_freqs, key=pair_freqs.get)

        merges.append(best_pair)

        # Merge all non-overlapping occurrences of the selected pair.
        new_word_freqs = {}

        for tokens, freq in word_freqs.items():
            merged_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and (tokens[i], tokens[i + 1]) == best_pair
                ):
                    merged_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged_tokens.append(tokens[i])
                    i += 1

            merged_tokens = tuple(merged_tokens)
            new_word_freqs[merged_tokens] = (
                new_word_freqs.get(merged_tokens, 0) + freq
            )

        word_freqs = new_word_freqs

    return merges