"""Evaluation helpers: pair metrics and fill-in-the-blank accuracy."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def binary_metrics(y_true, y_pred, y_score):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score))}


def fitb_accuracy(questions, score_fn, seed=0):
    """Share of fill-in-the-blank questions answered correctly.

    questions: dicts with 'question' (item_ids), 'answers' (4 item_ids), 'correct' (index).
    score_fn(question_items, candidate): scores of the candidate against each question item.
    Ties for the top score are broken at random.
    """
    rng = np.random.default_rng(seed)
    right = 0
    for q in questions:
        scores = np.array([np.mean(score_fn(q["question"], c)) for c in q["answers"]])
        tied_best = np.flatnonzero(scores == scores.max())
        right += int(rng.choice(tied_best) == q["correct"])
    return right / len(questions)