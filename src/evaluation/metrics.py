import re

import numpy as np

STOP = set("""a an the and or of to in on for with by at from as is are was were be been being this that these those it its
their there which who whom whose what when where how why shall will would may might can could should must not no any
all each such than then also into under over upon other if but so do does did has have had you your we our i he she
they them his her hereby herein thereof party parties agreement section""".split())


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def content_words(text):
    return [w for w in tokenize(text) if w not in STOP and len(w) > 2]


def distinct_n(texts, n):
    """Share of unique n-grams across all answers to one question (higher = more diverse)."""
    grams = []
    for t in texts:
        toks = tokenize(t)
        grams += [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def self_bleu(texts):
    """Mean BLEU of each answer against the others (LOWER = more diverse)."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    if len(texts) < 2:
        return 0.0
    smooth = SmoothingFunction().method1
    toks = [tokenize(t) for t in texts]
    scores = []
    for i, hyp in enumerate(toks):
        refs = [t for j, t in enumerate(toks) if j != i]
        if hyp and all(refs):
            scores.append(sentence_bleu(refs, hyp, weights=(0.5, 0.5), smoothing_function=smooth))
    return float(np.mean(scores)) if scores else 0.0


def grounding(answer, context_text):
    """Share of the answer's content words that appear in the retrieved excerpts or question."""
    words = content_words(answer)
    if not words:
        return 0.0
    ctx = set(content_words(context_text))
    return sum(w in ctx for w in words) / len(words)


def gold_recall(answer, gold_text):
    """Share of the gold clause's content words that the answer mentions."""
    gold = set(content_words(gold_text))
    if not gold:
        return 0.0
    ans = set(content_words(answer))
    return len(gold & ans) / len(gold)


def pairwise_length_spread(texts):
    lens = [len(tokenize(t)) for t in texts]
    return float(np.std(lens)) if lens else 0.0
