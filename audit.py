"""Audit the judge the way you would validate any rating instrument.

Five questions:
  1. Self-consistency  - does the judge agree with itself across passes? (test-retest)
  2. Agreement         - does it track the human consensus?
  3. Verbosity bias    - does answer length sway the score independent of quality?
  4. Group invariance  - does it score one group differently at equal quality? (DIF-style)
  5. Position bias      - in pairwise mode, does the first-shown answer win unfairly?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

import config


def wkappa(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")


def partial_corr(x, y, z):
    """Correlation of x and y after removing the linear effect of z from both."""
    def resid(a):
        A = np.vstack([z, np.ones_like(z)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return float(np.corrcoef(resid(x.astype(float)), resid(y.astype(float)))[0, 1])


def main() -> None:
    s = pd.read_csv(config.SINGLE)
    p = pd.read_csv(config.PAIRWISE)
    human_mean = np.round((s["human_1"] + s["human_2"]) / 2).astype(int)

    L = []
    L += ["AUDIT OF THE LLM JUDGE", "=" * 56, ""]

    L += ["1. SELF-CONSISTENCY (test-retest, quadratic-weighted kappa)"]
    L += [f"   judge pass-1 vs pass-2 : {wkappa(s['judge_pass_1'], s['judge_pass_2']):.3f}"]
    L += [f"   human-1 vs human-2     : {wkappa(s['human_1'], s['human_2']):.3f}   (reference)"]
    L += [""]

    L += ["2. AGREEMENT WITH HUMAN CONSENSUS"]
    L += [f"   weighted kappa : {wkappa(s['judge_pass_1'], human_mean):.3f}"]
    L += [f"   Pearson r      : {np.corrcoef(s['judge_pass_1'], human_mean)[0, 1]:.3f}"]
    L += [""]

    L += ["3. VERBOSITY BIAS (score ~ answer length | true quality, partial r)"]
    L += [f"   judge : {partial_corr(s['judge_pass_1'].values, s['answer_length'].values, s['true_quality'].values):+.3f}"]
    L += [f"   human : {partial_corr(human_mean.values, s['answer_length'].values, s['true_quality'].values):+.3f}   (reference)"]
    L += [""]

    L += ["4. GROUP INVARIANCE (mean of judge - true quality, by group)"]
    for g in sorted(s["group"].unique()):
        m = s["group"] == g
        L += [f"   group {g} : {(s.loc[m, 'judge_pass_1'] - s.loc[m, 'true_quality']).mean():+.2f}"]
    L += ["   a non-zero gap means the judge is not measurement-invariant across groups"]
    L += [""]

    L += ["5. POSITION BIAS (pairwise)"]
    first_wins = (p["picked_x_when_x_first"].mean() + (1 - p["picked_x_when_y_first"]).mean()) / 2
    flips = (p["picked_x_when_x_first"] != p["picked_x_when_y_first"]).mean()
    L += [f"   first-shown answer wins : {first_wins:.1%}   (50% = no position bias)"]
    L += [f"   verdict flips on swap   : {flips:.1%}"]

    report = "\n".join(L)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "audit.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
