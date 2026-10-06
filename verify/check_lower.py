"""Constants of the memory lower bound (Theorem 7.3; explicit form Theorem J.1): leakage <= (32/15) eps/n and
n a <= 6.4 Q + 13.52 + 0.134 log2 n give Q >= (L - 13.6 - 0.14 log2 n)/7.4, tested on a grid of (eps, n, Q); h2(1/16) <= 0.34.
Also evaluates the Houghton exponent beta_H/(1 + beta_H) = 1/(2 log2 107 + 2) of Remark 8.14.
Run: python -u check_lower.py
"""
import json
import numpy as np


def h2(x):
    return 0.0 if x in (0.0, 1.0) else float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))


eps_max = 1 / 16
c_leak = 2 / (1 - eps_max)                      # leakage factor in units of eps/n
coef_Q = 1 + 3 * c_leak                         # coefficient of Q in S + n a
const = c_leak * (6 + h2(eps_max))              # constant term
coef_log = c_leak * eps_max                     # coefficient of log2 n
print(f"leakage factor 2/(1-1/16) = {c_leak:.6f} (= 32/15)")
print(f"S + n a <= {coef_Q:.4f} Q + {const:.4f} + {coef_log:.4f} log2 n")
print(f"h2(1/16) = {h2(eps_max):.5f}")

# the chain holds for every eps <= 1/16, n >= 1, Q >= 0 (worst case B = 3Q + 6, S = Q)
worst = -np.inf
for eps in np.linspace(1e-6, eps_max, 200):
    for n in [1, 2, 10, 10 ** 3, 10 ** 6, 10 ** 12]:
        for Q in [0.0, 1.0, 10.0, 1e3, 1e6]:
            na = 2 * (3 * Q + 6 + h2(eps) + eps * np.log2(n)) / (1 - eps)
            lhs = Q + na
            rhs = 7.4 * Q + 13.6 + 0.14 * np.log2(n)
            worst = max(worst, lhs - rhs)
print(f"max over grid of (Q + n a) - (7.4 Q + 13.6 + 0.14 log2 n) = {worst:.4f} (must be <= 0)")

beta_H = 1 / (2 * np.log2(107) + 1)
results = {
    "c_leak": c_leak, "coef_Q": coef_Q, "const": const, "coef_log": coef_log, "h2_1_16": h2(eps_max),
    "chain_margin": worst,
    "beta_H": beta_H, "beta_H_rank_rate": beta_H / (1 + beta_H),
}
print("beta_H = %.5f, beta_H/(1+beta_H) = %.5f (Remark 8.14: 1/(2 log2 107 + 2))" %
      (beta_H, beta_H / (1 + beta_H)))
print("RESULTS_JSON " + json.dumps(results))
