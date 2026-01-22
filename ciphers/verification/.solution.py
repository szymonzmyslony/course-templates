def decipher_corpus(clear_corpus, ciphered_corpus):
    """
    Self-contained decipherer based on the provided notebook logic.

    Args:
        clear_corpus:   list[str]  (reference "clear" lines, already lowercased/stripped preferred)
        ciphered_corpus:list[str]  (ciphered lines to decipher, already lowercased/stripped preferred)

    Returns:
        list[str]: deciphered lines (unknown chars become '🦡')
    """
    # Imports kept inside to make the function self-contained
    from gensim.models import Word2Vec as Letter2Vec
    from collections import defaultdict as dd

    # ---- normalize input (match notebook behavior) ----
    corpus_clear = [str(line).strip().lower() for line in clear_corpus]
    corpus_ciphered = [str(line).strip().lower() for line in ciphered_corpus]

    # ---- 1) Train Letter2Vec on ciphered corpus ----
    model = Letter2Vec(
        sentences=corpus_ciphered,
        vector_size=20,
        window=2,
        min_count=10,
        workers=4,
    )
    keys = model.wv.index_to_key

    # ---- 2) Find "good triples" and build correcting_table ----
    stats = dd(int)
    for k in keys:
        Rloc = [k]
        Rloc += [k0 for (k0, _) in model.wv.most_similar(k, topn=2)]
        Rloc = tuple(sorted(Rloc))
        stats[Rloc] += 1

    good_triples = set()
    for k, v in stats.items():
        if v == 3:
            good_triples.update(k)

    correcting_table = {}
    for t in stats:
        for a in t:
            if a in good_triples:
                correcting_table[a] = t[0]

    # ---- 3) Apply correcting_table to ciphered corpus ----
    corpus_ciphered_unified = []
    for line in corpus_ciphered:
        unified_line = []
        for ch in line:
            unified_line.append(correcting_table.get(ch, ch))
        corpus_ciphered_unified.append("".join(unified_line))

    # ---- 4) N-gram statistics helpers ----
    def normalize_dist(d):
        s = float(sum(d.values())) or 1.0
        for x in list(d.keys()):
            d[x] /= s
        return d

    def stats_for_corpus(corpus, N):
        st = dd(int)
        for line in corpus:
            if len(line) < N:
                continue
            for i in range(len(line) - N + 1):
                b = tuple(line[i : i + N])
                st[b] += 1
        return normalize_dist(st)

    stats_clear1 = stats_for_corpus(corpus_clear, 1)
    stats_clear3 = stats_for_corpus(corpus_clear, 3)

    stats_dirty1 = stats_for_corpus(corpus_ciphered_unified, 1)
    stats_dirty3 = stats_for_corpus(corpus_ciphered_unified, 3)

    # ---- 5) Build trigram candidate sets ----
    def add_most_frequent(trigrams, st3):
        most_freq_val = dd(float)
        most_freq = {}
        for t in st3:
            for a in t:
                if st3[t] > most_freq_val[a]:
                    most_freq_val[a] = st3[t]
                    most_freq[a] = t
        for t in most_freq.values():
            trigrams.add(t)

    top_trigrams_clear = set(
        sorted(stats_clear3, key=stats_clear3.get, reverse=True)[:400]
    )
    top_trigrams_dirty = set(
        sorted(stats_dirty3, key=stats_dirty3.get, reverse=True)[:400]
    )

    add_most_frequent(top_trigrams_clear, stats_clear3)
    add_most_frequent(top_trigrams_dirty, stats_dirty3)

    # ---- 6) Define trigram distance and infer mapping R ----
    def prob_dist(p1, p2):
        # NOTE: name collision in notebook; this is the *distance* version
        if p1 == 0 and p2 == 0:
            return 0.0
        m = max(p1, p2)
        return abs(p1 - p2) / m if m else 0.0

    def dist(tc, td):
        unigram_prob_c = [stats_clear1.get((x,), 0.0) for x in tc]
        unigram_prob_d = [stats_dirty1.get((x,), 0.0) for x in td]
        res = sum(prob_dist(cc, dd_) for cc, dd_ in zip(unigram_prob_c, unigram_prob_d))
        return res + prob_dist(stats_clear3.get(tc, 0.0), stats_dirty3.get(td, 0.0))

    aux = []
    for c in top_trigrams_clear:
        for d in top_trigrams_dirty:
            aux.append((dist(c, d), c, d))
    aux.sort(key=lambda x: x[0])

    R = {}
    for _, t1, t2 in aux:
        correct = 0
        for a, b in zip(t1, t2):
            if b not in R or R[b] == a:
                correct += 1
        if correct == 3:
            for a, b in zip(t1, t2):
                R[b] = a

    # ---- 7) Apply mapping to decode ----
    def change(mapping, line):
        out = []
        for ch in line:
            if ch in mapping:
                out.append(mapping[ch])
            else:
                out.append("🦡")  # unknown
        return "".join(out)

    return [change(R, line) for line in corpus_ciphered_unified]
