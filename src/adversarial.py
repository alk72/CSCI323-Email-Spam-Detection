import numpy as np

LEET = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5"})


def obfuscate_word(word: str, rng) -> str:
    if rng.random() < 0.5:
        return word.translate(LEET)
    return ".".join(word)


def obfuscate_text(text: str, rate: float, rng, targets=None) -> str:
    """Disguise each target word with probability `rate` (all words if targets is None)."""
    words = text.split()
    for i, w in enumerate(words):
        eligible = w.lower().strip(".,!?:;\"'()") in targets if targets is not None else (len(w) > 3 and w.isalpha())
        if eligible and rng.random() < rate:
            words[i] = obfuscate_word(w, rng)
    return " ".join(words)


def good_word_attack(text: str, good_words, n_words: int, rng) -> str:
    if n_words == 0 or len(good_words) == 0:
        return text
    return text + " " + " ".join(rng.choice(good_words, size=n_words))


def _word_weights(pipe):
    vocab = np.array(pipe.named_steps["vec"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_log_prob_"):
        weight = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    else:
        weight = clf.coef_.ravel()
    return vocab, weight


def top_ham_words(pipe, k=200):
    vocab, weight = _word_weights(pipe)
    return vocab[np.argsort(weight)[:k]].tolist()


def top_spam_words(pipe, k=200):
    vocab, weight = _word_weights(pipe)
    return vocab[np.argsort(-weight)[:k]].tolist()
