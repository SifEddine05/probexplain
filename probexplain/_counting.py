"""Exact model counting on orthogonal DNFs (shared by the UAI 2023 and k-pseudo algorithms).

Counts are exact Python integers, so the results (and the tie-breaking of the greedy algorithms)
are the same as in the original code, and nothing overflows when the number of Boolean variables
is large.
"""
from fractions import Fraction
import math


def M_s(phi, t, n):
    """Number of models of ``phi ∧ t`` over n Boolean variables (phi: orthogonal DNF, t: term)."""
    t_set = set(t)
    t_neg = {-c for c in t}
    w = 0
    for cl in phi:
        cl_set = set(cl)
        if cl_set & t_neg:
            continue
        w += 1 << (n - len(cl_set | t_set))
    return w


def W_s(phi, t, n):
    """Probability that phi holds when the literals of t are fixed and the others are uniform."""
    return M_s(phi, t, n) / (1 << (n - len(set(t))))


def log(v):
    """Natural log of a (possibly huge) int or Fraction."""
    if isinstance(v, Fraction):
        return math.log(v.numerator) - math.log(v.denominator)
    return math.log(v)
