"""SQuAD/TyDi-style QA answer metrics (Exact Match + token F1).

Used by the SVQ RAG eval (``scripts/run_svq_rag_eval.py``) to score a generated
answer against one or more acceptable gold answers, plus unanswerable handling.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = {ord(c): None for c in string.punctuation}

# Strings that indicate an "unanswerable" gold or prediction.
NO_ANSWER_MARKERS = {"", "no answer", "none", "unanswerable", "n/a", "no_answer"}


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace (SQuAD-style, unicode-safe)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def is_no_answer(s: str) -> bool:
    return normalize_answer(s) in NO_ANSWER_MARKERS


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, golds: Sequence[str]) -> tuple[float, float]:
    """Best (EM, F1) of ``prediction`` over acceptable ``golds``.

    Unanswerable questions (empty gold list or all no-answer markers) score 1.0
    when the prediction is also a no-answer, else 0.0.
    """
    gold_list = [g for g in golds if g is not None]
    if not gold_list or all(is_no_answer(g) for g in gold_list):
        no_ans = 1.0 if is_no_answer(prediction) else 0.0
        return no_ans, no_ans
    em = max(exact_match(prediction, g) for g in gold_list)
    f1 = max(token_f1(prediction, g) for g in gold_list)
    return em, f1
