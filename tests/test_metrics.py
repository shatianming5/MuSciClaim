from musciclaim.metrics.classification import compute_classification_metrics
from musciclaim.metrics.localization import compute_localization_metrics
from musciclaim.metrics.sensitivity import compute_flip_rates


def test_classification_perfect():
    gold = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
    pred = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
    m = compute_classification_metrics(
        gold=gold,
        pred=pred,
        labels=["SUPPORT", "NEUTRAL", "CONTRADICT"],
        valid_only=False,
    )
    assert m.macro_f1 == 1.0


def test_classification_invalid_counts_as_error_in_e2e():
    gold = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
    pred = [None, "CONTRADICT", "NEUTRAL"]
    m = compute_classification_metrics(
        gold=gold,
        pred=pred,
        labels=["SUPPORT", "NEUTRAL", "CONTRADICT"],
        valid_only=False,
    )
    assert 0.0 <= m.macro_f1 < 1.0


def test_localization_set_f1_basic():
    gold = [["Panel A"], ["Panel B"], []]
    pred = [["Panel A"], [], []]
    m = compute_localization_metrics(gold_sets=gold, pred_sets=pred)
    assert 0.0 <= m.f1 <= 1.0


def test_flip_rates():
    records = [
        {"base_claim_id": "b1", "label_gold": "SUPPORT", "label_pred": "SUPPORT"},
        {"base_claim_id": "b1", "label_gold": "CONTRADICT", "label_pred": "CONTRADICT"},
        {"base_claim_id": "b2", "label_gold": "SUPPORT", "label_pred": "SUPPORT"},
        {"base_claim_id": "b2", "label_gold": "CONTRADICT", "label_pred": "SUPPORT"},
    ]
    m = compute_flip_rates(records=records, allowed_labels={"SUPPORT", "CONTRADICT", "NEUTRAL"})
    assert m.num_pairs_total == 2
    assert m.num_pairs_valid == 2
    assert m.strict_flip_rate == 0.5
