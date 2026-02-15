from musciclaim.metrics.bootstrap import paired_bootstrap_ci_diff


def test_paired_bootstrap_ci_diff_smoke():
    gold = ["a", "b", "c", "a", "b"]
    pred_a = ["a", "b", "c", "a", "b"]
    pred_b = ["a", "c", "c", "b", "b"]

    def acc(g, p):
        return sum(1 for gg, pp in zip(g, p, strict=True) if gg == pp) / float(len(g))

    ci = paired_bootstrap_ci_diff(
        gold=gold,
        pred_a=pred_a,
        pred_b=pred_b,
        metric_fn=acc,
        iters=500,
        seed=123,
    )

    # Point estimate sanity.
    assert ci.n == len(gold)
    assert ci.iters == 500
    assert ci.seed == 123

    expected = acc(gold, pred_a) - acc(gold, pred_b)
    assert abs(ci.diff - expected) < 1e-12

    # Interval sanity.
    assert ci.ci95_low <= ci.ci95_high
    assert -1.0 <= ci.ci95_low <= 1.0
    assert -1.0 <= ci.ci95_high <= 1.0
