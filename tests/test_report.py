from bloom_eval.report import compare, two_prop_p, wilson


def run(rid, passes, fails, name="color"):
    results = [{"task_id": "t", "sample": i, "checks": [{"name": name, "passed": True, "score": 1, "detail": {}}]} for i in range(passes)]
    results += [{"task_id": "t", "sample": 100 + i, "checks": [{"name": name, "passed": False, "score": 0, "detail": {}}]} for i in range(fails)]
    return {"run_id": rid, "skill": {"skill_id": rid}, "results": results}


def test_wilson_bounds():
    lo, hi = wilson(9, 10)
    assert 0.5 < lo < 0.9 < hi <= 1.0
    assert wilson(0, 0) == (0.0, 1.0)


def test_two_prop_p_identical_is_one():
    assert two_prop_p(10, 20, 10, 20) > 0.99


def test_compare_flags_real_regression_but_not_noise():
    base = run("a", 45, 5)           # 90%
    worse = run("b", 30, 20)         # 60%
    noise = run("c", 43, 7)          # 86%
    assert compare(base, worse)[0]["regression"] is True
    assert compare(base, noise)[0]["regression"] is False
