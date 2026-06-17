"""Build the audit datasets.

Two experiments:
  single_answer.csv  - each item has a true quality, an answer length, a group
                       flag, two human ratings, and two independent judge passes
                       (so we can measure the judge's test-retest reliability).
  pairwise.csv       - head-to-head comparisons scored in both presentation
                       orders, so we can measure position bias.

The judge is simulated to exhibit documented behaviors (verbosity bias, a group
tilt, and run-to-run inconsistency). To audit a *real* judge instead, replace
`judge_score` / `judge_pick_first` with calls to the model and keep everything
downstream unchanged.
"""
import numpy as np
import pandas as pd

import config

RNG = np.random.default_rng(config.RANDOM_STATE)


def _clamp(x: float) -> int:
    return int(np.clip(round(x), config.SCORE_MIN, config.SCORE_MAX))


def judge_score(true_quality, length, group, noise):
    """Simulated single-answer judge. Score tracks true quality but is pulled by
    answer length (verbosity bias) and tilted against group B, with a noise term
    that makes repeat scorings disagree (limited self-consistency)."""
    raw = (true_quality
           + config.VERBOSITY_BETA * (length - 180)
           + (config.GROUP_TILT if group == "B" else 0.0)
           + noise)
    return _clamp(raw)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def judge_pick_first(q_first, q_second):
    """Simulated pairwise judge: probability of choosing the FIRST-shown answer.
    The POSITION_PULL term biases the judge toward whatever it sees first."""
    return _sigmoid(0.9 * (q_first - q_second) + config.POSITION_PULL)


def build_single():
    n = config.N_SINGLE
    true_q = RNG.integers(config.SCORE_MIN, config.SCORE_MAX + 1, n)
    length = (RNG.normal(180, 70, n) + 12 * true_q).clip(40, 500).round()
    group = RNG.choice(["A", "B"], n)

    human_1 = [_clamp(true_q[i] + RNG.normal(0, 0.5)) for i in range(n)]
    human_2 = [_clamp(true_q[i] + RNG.normal(0, 0.5)) for i in range(n)]
    judge_1 = [judge_score(true_q[i], length[i], group[i], RNG.normal(0, config.INCONSISTENCY_SD)) for i in range(n)]
    judge_2 = [judge_score(true_q[i], length[i], group[i], RNG.normal(0, config.INCONSISTENCY_SD)) for i in range(n)]

    return pd.DataFrame({"true_quality": true_q, "answer_length": length.astype(int),
                         "group": group, "human_1": human_1, "human_2": human_2,
                         "judge_pass_1": judge_1, "judge_pass_2": judge_2})


def build_pairwise():
    m = config.N_PAIRWISE
    q_x = RNG.integers(config.SCORE_MIN, config.SCORE_MAX + 1, m)
    q_y = RNG.integers(config.SCORE_MIN, config.SCORE_MAX + 1, m)
    # order 1: X is shown first; order 2: Y is shown first (same pair, swapped)
    pick_x_order1 = RNG.random(m) < judge_pick_first(q_x, q_y)
    pick_x_order2 = ~(RNG.random(m) < judge_pick_first(q_y, q_x))
    return pd.DataFrame({"q_x": q_x, "q_y": q_y,
                         "picked_x_when_x_first": pick_x_order1.astype(int),
                         "picked_x_when_y_first": pick_x_order2.astype(int)})


if __name__ == "__main__":
    config.SINGLE.parent.mkdir(parents=True, exist_ok=True)
    build_single().to_csv(config.SINGLE, index=False)
    build_pairwise().to_csv(config.PAIRWISE, index=False)
    print(f"Wrote {config.SINGLE} and {config.PAIRWISE}")
