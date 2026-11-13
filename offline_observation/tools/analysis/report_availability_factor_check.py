"""Deterministic report-factor calculation; no dataset or author reproduction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def log_detection_factor(present, probabilities, available):
    """Eligible independent Bernoulli report terms for one specified cluster."""
    terms = []
    for source, probability in enumerate(probabilities):
        if not 0 < probability < 1:
            raise ValueError('fixed probabilities must be strictly between zero and one')
        if not available[source]:
            if source in present:
                raise ValueError('an unavailable source cannot supply an observed report')
            continue
        terms.append(math.log(probability if source in present else 1-probability))
    return math.fsum(terms)


def log_merge_split_ratio(probabilities, available):
    merge = log_detection_factor({0, 1}, probabilities, available)
    split = (log_detection_factor({0}, probabilities, available)
             + log_detection_factor({1}, probabilities, available))
    return math.log(.2) + merge-split


def calculate():
    original = log_merge_split_ratio([.5, .5], [True, True])
    rows = []
    for p3 in [.1, .25, .5, .75, .9]:
        bad = log_merge_split_ratio([.5, .5, p3], [True, True, True])
        adapted = log_merge_split_ratio([.5, .5, p3], [True, True, False])
        analytic = .8/(1-p3)
        if not (math.isclose(math.exp(original), .8, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(math.exp(bad), analytic, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(adapted, original, rel_tol=1e-12, abs_tol=1e-12)):
            raise ArithmeticError('fixed numerical/analytical report factors disagree')
        rows.append(dict(p3=p3, original_log_ratio=original,
                         unavailable_but_counted_log_ratio=bad,
                         conditioned_unavailable_log_ratio=adapted,
                         available_missing_negative_control_log_ratio=bad,
                         original_ratio=math.exp(original),
                         unavailable_but_counted_ratio=math.exp(bad),
                         conditioned_unavailable_ratio=math.exp(adapted),
                         available_missing_negative_control_ratio=math.exp(bad),
                         ordering_flip=(original < 0 < bad)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    run = dict(status='RUNNING', started_at=datetime.now(timezone.utc).isoformat(),
               source_idea_id='IDEA-REPORT-AVAILABILITY-FACTOR-20261029',
               plan_version='1.0', dataset_read=False, synthetic_sampling=False,
               model_forwards=0, optimizer_updates=0, test_opened=False,
               scope='deterministic conditional-factor calculation; not MTAD effect')
    code = 1
    try:
        rows = calculate()
        (args.output/'summary.json').write_text(json.dumps(dict(
            rows=rows, row_count=len(rows), arithmetic_checks_passed=True,
            target_applicability='unknown availability; current056 has no such factor',
            author_reproduction=False, mtad_effect_measured=False), indent=2, allow_nan=False)+'\n')
        run.update(status='COMPLETED', completed_conditions=len(rows))
        code = 0
    except Exception as exc:
        run.update(status='FAILED', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        run.update(ended_at=datetime.now(timezone.utc).isoformat(),
                   elapsed_seconds=time.monotonic()-start, exit_code=code)
        (args.output/'run.json').write_text(json.dumps(run, indent=2, allow_nan=False)+'\n')
        (args.output/'exit_code.txt').write_text(str(code)+'\n')


if __name__ == '__main__':
    main()
