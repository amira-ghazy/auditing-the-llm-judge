"""Configuration for the LLM-judge audit.

The bias magnitudes below define a *simulated* judge that reproduces behaviors
documented in the LLM-as-judge literature (Zheng et al., 2023, "Judging LLM-as-a-
Judge with MT-Bench and Chatbot Arena," which reports both position bias and
verbosity bias). The point of the project is not the simulation; it is that
standard psychometric audits detect these behaviors. Swap `judge_score` in
simulate.py for a live model call to run the same audit on a real judge.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SINGLE = ROOT / "data" / "single_answer.csv"
PAIRWISE = ROOT / "data" / "pairwise.csv"
RESULTS = ROOT / "results"

SCORE_MIN, SCORE_MAX = 1, 5
RANDOM_STATE = 7

N_SINGLE = 300        # items scored one answer at a time
N_PAIRWISE = 400      # head-to-head comparisons

# documented-bias knobs for the simulated judge
VERBOSITY_BETA = 0.0045   # score gain per token of answer length
GROUP_TILT = -0.35        # systematic penalty applied to group B at equal quality
INCONSISTENCY_SD = 0.6    # noise term -> imperfect test-retest reliability
POSITION_PULL = 0.8       # log-odds pull toward whichever answer is shown first
