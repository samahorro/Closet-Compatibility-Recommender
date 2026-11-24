from ml.evaluate import fitb_accuracy


def test_fitb_accuracy_picks_highest_score():
    q = [{"question": ["a", "b"], "answers": ["w", "x", "y", "z"], "correct": 2}]
    assert fitb_accuracy(q, lambda question, cand: [1.0 if cand == "y" else 0.0]) == 1.0
