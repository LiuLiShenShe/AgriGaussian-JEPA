import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import evaluation_due


def test_evaluation_waits_for_opacity_reset_recovery():
    # DefaultStrategy resets opacity AFTER the step. Evaluating immediately
    # after 3000 yields a temporarily transparent object, not model quality.
    kwargs = dict(every=3000, total_steps=30000, reset_every=3000,
                  refine_stop=15000, recovery_steps=179)
    assert not evaluation_due(3000, **kwargs)
    assert not evaluation_due(3178, **kwargs)
    assert evaluation_due(3179, **kwargs)
    assert evaluation_due(6179, **kwargs)
    assert evaluation_due(15000, **kwargs)  # resets have stopped
    assert evaluation_due(18000, **kwargs)
    assert evaluation_due(29999, **kwargs)


def test_short_run_evaluates_last_step_even_without_a_full_interval():
    kwargs = dict(every=200, total_steps=200, reset_every=3000,
                  refine_stop=160, recovery_steps=179)
    assert not evaluation_due(0, **kwargs)
    assert evaluation_due(199, **kwargs)
