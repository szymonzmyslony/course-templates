import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import random
import math


def decipher_corpus(clear_corpus, ciphered_corpus):
    """
    Deciphers the ciphered_corpus using statistics from clear_corpus.
    Strategy:
    1. Letter2Vec to embed symbols based on context.
    2. KMeans to group the 3 symbols for 'a', 'b', etc. together.
    3. Frequency matching to assign clusters to likely letters.
    4. Bigram-based hill climbing to correct the mapping.
    """

    # --- 1. Preprocessing & Vocabulary Building ---

    # Flatten clear corpus to get alphabet and counts
    clear_text_joined = "".join(clear_corpus)
    clear_counter = Counter(clear_text_joined)
    alphabet = sorted(list(clear_counter.keys()))
    num_chars = len(alphabet)

    # Build Bigram Statistics for Clear Text (for scoring)
    bigram_counts = defaultdict(lambda: 0)
    for line in clear_corpus:
        for c1, c2 in zip(line, line[1:]):
            bigram_counts[(c1, c2)] += 1

    # Normalize bigrams to log-probabilities with Laplace smoothing
    total_bigrams = sum(bigram_counts.values())
    bigram_log_probs = defaultdict(lambda: -15.0)  # Default low probability
    vocab_size = len(alphabet)

    for (c1, c2), count in bigram_counts.items():
        # Smoothed probability: log((count + 1) / (total + vocab^2))
        prob = (count + 1) / (total_bigrams + vocab_size**2)
        bigram_log_probs[(c1, c2)] = math.log(prob)

    # --- 2. Train Embeddings (Letter2Vec) ---

    # Tokenize cipher corpus for Gensim (list of list of symbols)
    cipher_sentences = [list(line) for line in ciphered_corpus]

    # Train Word2Vec on symbols
    # Window=3 captures immediate neighbors. min_count=1 ensures all symbols are included.
    model = Word2Vec(
        sentences=cipher_sentences,
        vector_size=30,
        window=3,
        min_count=1,
        workers=4,
        epochs=20,
        sg=1,
    )

    # Get all unique symbols in cipher text
    cipher_symbols = list(model.wv.index_to_key)
    vectors = np.array([model.wv[s] for s in cipher_symbols])

    # --- 3. Clustering ---

    # We know there are `num_chars` (n) bags.
    # We cluster the symbols into `num_chars` clusters.
    # Ideally, each cluster contains the 3 symbols for one specific letter.
    kmeans = KMeans(n_clusters=num_chars, n_init=10, random_state=42)
    kmeans.fit(vectors)
    labels = kmeans.labels_

    # Map each symbol to a Cluster ID (0 to n-1)
    symbol_to_cluster_id = {sym: label for sym, label in zip(cipher_symbols, labels)}

    # Handle symbols that might not be in the model (extremely rare edge case)
    # Assign them to a random cluster or closest one.
    # Here we assume all symbols appeared at least once.

    # --- 4. Initial Mapping by Frequency ---

    # Calculate Cluster Frequencies in Cipher Text
    cluster_counter = Counter()
    for line in ciphered_corpus:  # Count directly from corpus just to be safe
        for char in line:
            if char in symbol_to_cluster_id:
                cluster_counter[symbol_to_cluster_id[char]] += 1

    # Sort clear letters by frequency
    sorted_clear_chars = [pair[0] for pair in clear_counter.most_common()]

    # Sort cipher clusters by frequency
    sorted_clusters = [pair[0] for pair in cluster_counter.most_common()]

    # If counts mismatch (some clusters might have 0 count if extremely rare, though unlikely), pad
    if len(sorted_clusters) < len(sorted_clear_chars):
        missing = set(range(num_chars)) - set(sorted_clusters)
        sorted_clusters.extend(list(missing))

    # Create initial mapping: Rank 1 Cluster -> Rank 1 Letter
    cluster_to_char_map = {}
    for i in range(min(len(sorted_clusters), len(sorted_clear_chars))):
        cluster_to_char_map[sorted_clusters[i]] = sorted_clear_chars[i]

    # --- 5. Refinement: Hill Climbing with Bigrams ---

    # Create a smaller sample for optimization to speed up the loop
    # We convert the sample to a list of Cluster IDs to avoid looking up symbols constantly
    sample_size = min(2000, len(ciphered_corpus))
    validation_lines_ids = []

    for line in ciphered_corpus[:sample_size]:
        ids = [symbol_to_cluster_id.get(c, -1) for c in line]
        # Filter out unknown symbols (-1) if any
        validation_lines_ids.append([x for x in ids if x != -1])

    def score_mapping(current_map):
        """Calculate log likelihood of the validation set under current mapping."""
        total_score = 0.0
        for line_ids in validation_lines_ids:
            # Convert cluster IDs to proposed text
            text = [current_map[cid] for cid in line_ids]
            # Sum bigram logs
            for i in range(len(text) - 1):
                total_score += bigram_log_probs[(text[i], text[i + 1])]
        return total_score

    # Initial Score
    current_score = score_mapping(cluster_to_char_map)

    # Optimization Loop
    # Try swapping assignments to improve bigram probability
    iterations = 3000
    distinct_clusters = list(cluster_to_char_map.keys())

    for k in range(iterations):
        # Pick two random clusters to swap
        c1, c2 = random.sample(distinct_clusters, 2)

        char1 = cluster_to_char_map[c1]
        char2 = cluster_to_char_map[c2]

        # Swap
        cluster_to_char_map[c1] = char2
        cluster_to_char_map[c2] = char1

        new_score = score_mapping(cluster_to_char_map)

        if new_score > current_score:
            current_score = new_score
            # Keep the swap
        else:
            # Revert the swap
            cluster_to_char_map[c1] = char1
            cluster_to_char_map[c2] = char2

    # --- 6. Final Deciphering ---

    deciphered_lines = []
    for line in ciphered_corpus:
        decoded_chars = []
        for symbol in line:
            if symbol in symbol_to_cluster_id:
                cluster_id = symbol_to_cluster_id[symbol]
                decoded_chars.append(cluster_to_char_map[cluster_id])
            else:
                # Fallback for unknown symbol (shouldn't happen with min_count=1)
                decoded_chars.append(" ")
        deciphered_lines.append("".join(decoded_chars))

    return deciphered_lines
