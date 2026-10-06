"""The re-folding chain, Lemma H.14. A: with integer slot counts the re-folding injection reproduces the chain of Lemma H.14(a)
exactly. B: on the per-coordinate step laws of Houghton's channel the RMS distance matches |delta|/sqrt(2 gamma), as stated after
Lemma H.14, and the loss at the radius of Lemma H.14(b) is negligible. Part C prints a supplementary device estimate.
Run: python -u check_refold.py   (about 15 s)
"""
import json
import math

import numpy as np
from scipy.special import gammaln

res = {}

# ------------------------------------------------------------------ Part A: integer slots
worst, cases = 0.0, 0
for m, gamma in [(50, 0.02), (101, 0.013), (400, 0.002), (37, 0.11), (1000, 0.0007)]:
    mp = math.ceil((1 + gamma) * m)
    gt = (mp - m) / m
    J = int(1.5 / gt) + 3
    for j in range(J):
        for s in (0, 1):
            ranks = 2 * j * m + 2 * np.arange(m) + s
            tgt = ranks // (2 * mp)
            if j * gt <= 1:
                down = np.mean(tgt == j - 1) if j >= 1 else 0.0
                stay = np.mean(tgt == j)
                worst = max(worst, abs(down - j * gt), abs(stay - (1 - j * gt)))
            else:
                worst = max(worst, float(np.any(tgt > j - 1)))
            cases += 1
    assert gt >= gamma
res["slots_exact_err"] = worst
res["slots_cases"] = cases
print(f"Part A: integer-slot injection, {cases} (pair, position) cases: max |fraction - chain| = {worst:.1e}")

# ------------------------------------------------------------------ Part B: fluid chain on Houghton's coordinates
LAWS = {"coord1": {-2: .07692, -1: .28846, 0: .57692, 1: .05769},
        "coord2": {-1: .10096, 0: .87499, 1: .02404}}
gamma, n, eta = 0.002, 3000, 1e-4
c = 1 + gamma


def kernel(J):
    K = np.zeros((J, J)); over = np.zeros(J)
    tb = 2 * c * np.arange(J + 1)
    for j in range(J):
        a, b = 2 * j, 2 * j + 2
        i0 = int(np.floor(a / (2 * c)))
        for i in range(max(i0 - 1, 0), min(i0 + 3, J)):
            K[j, i] += max(0.0, min(b, tb[i + 1]) - max(a, tb[i])) / 2
        over[j] = max(0.0, b - max(a, tb[J])) / 2
    return K, over


def step(state, delta, J):
    """class-uniformized step on pair ranks: pair at distance d = j + 1/2 goes to distance |d + s delta|"""
    new = np.zeros(J + abs(delta) + 2)
    j = np.arange(J)
    for s in (1, -1):
        X = np.abs(j + 0.5 + s * delta)
        np.add.at(new, (X - 0.5).astype(int), state / 2)
    return new


def run(law, forced, J, start):
    K, over = kernel(J + 8)
    st, lost = start.copy(), 0.0
    for _ in range(n):
        def padded(dl):
            s_ = step(st, dl, J)
            out = np.zeros(J + 8); k = min(len(s_), J + 8); out[:k] = s_[:k]
            return out, float(s_[k:].sum())
        if forced is None:
            src = np.zeros(J + 8)
            for dl, w in law.items():
                p_, extra = padded(dl); src += w * p_; lost += w * extra
        else:
            src, extra = padded(forced); lost += extra
        lost += float((src * over).sum())
        out_ = src @ K
        st = out_[:J]
        lost += float(out_[J:].sum())
    dd = np.arange(J) + 0.5
    return float(np.sqrt((st * dd ** 2).sum() / max(st.sum(), 1e-300))), lost


J0 = int(1 / (2 * gamma))
K, _ = kernel(J0 + 2)
err = 0.0
for j in range(J0):
    target = np.zeros(J0 + 2); target[j] = 1 - j * gamma
    if j >= 1:
        target[j - 1] = j * gamma
    err = max(err, float(np.abs(K[j] - target).max()))
res["fluid_kernel_err"] = err
print(f"Part B: fluid kernel = chain to {err:.1e} (pairs j <= {J0})")
ratios, lost_B, lost_tight, rms_all, pred_all = [], [], [], [], []
for name, law in LAWS.items():
    Ed2 = sum(w * dl * dl for dl, w in law.items())
    dmax = max(abs(dl) for dl in law)
    pred_typ = math.sqrt(Ed2 / (2 * gamma))
    L = 20.0
    for _ in range(50):
        L = math.log(n / eta) + math.log(8 * dmax ** 2 / (gamma * L))
    dd = np.arange(4000) + 0.5
    start = np.exp(-dd ** 2 / (2 * pred_typ ** 2)); start /= start.sum()
    R0 = float(dd[np.searchsorted(np.cumsum(start), 1 - eta / 2)])
    xB = R0 + 2 * dmax * math.sqrt(2 * L / gamma) + 2 * dmax + 1
    Jb = int(xB)
    rms, lost = run(law, None, Jb, start[:Jb] / start[:Jb].sum())
    ratios.append(rms / pred_typ); lost_B.append(lost); rms_all.append(rms); pred_all.append(pred_typ)
    print(f"  {name}: typical RMS {rms:.2f} vs {pred_typ:.2f}; radius of Lemma H.14(b) {xB:.1f}; typical loss {lost:.1e}")
    for dl in sorted({abs(x) for x in law if x != 0}):
        pred = math.sqrt(dl * dl / (2 * gamma))
        rms_f, lost_f = run(law, dl, Jb, start[:Jb] / start[:Jb].sum())
        Jt = int(4.5 * pred)
        _, lost_f2 = run(law, dl, Jt, start[:Jt] / start[:Jt].sum())
        ratios.append(rms_f / pred); lost_B.append(lost_f); lost_tight.append(lost_f2)
        rms_all.append(rms_f); pred_all.append(pred)
        print(f"    forced |delta| = {dl}: RMS {rms_f:.2f} vs {pred:.2f}; loss at radius of Lemma H.14(b) {lost_f:.1e}, "
              f"at 4.5 x RMS {lost_f2:.2e}")
res["rms_ratios"] = ratios                      # coord1 typical, forced 1, forced 2, coord2 typical, forced 1
res["rms"] = rms_all
res["rms_pred"] = pred_all
res["loss_at_radius_max"] = max(lost_B)
res["loss_tight"] = lost_tight

# ------------------------------------------------------------------ Part C: genuine Houghton device, main terms
eps, k = 0.01, 2
LOG2 = math.log(2)


def device_Q(nr, g):
    eta_ = min(eps ** 2 / (64 * nr), 1 / (4 * nr ** 3))
    L_ = math.log(k * nr * (1 + 8 / g) / eta_)
    if g * L_ > 1 / 8:                          # Lemma H.14(b) needs gamma L <= 1/(2 delta_max^2), delta_max = 2
        return math.inf
    x = 0.5 + 2 * 2 * math.sqrt(2 * L_ / g) + 2 * 2 + 0.5
    M = 2 * math.ceil(x) + 26
    return gammaln(2 * M + 1) / LOG2 + 2 * math.log2(M) + 2 * nr * math.log2(1 + g)


cost_ratio = []
for nr in [1e8, 1e10, 1e12, 1e14, 1e16]:
    LF = 3.69 * nr ** (1 / 3) * math.log2(nr) ** (2 / 3)
    gs = np.exp(np.linspace(math.log(1e-14), math.log(1e-2), 4000))
    Q = min(device_Q(nr, g) for g in gs)
    cost_ratio.append(Q / LF)
    print(f"Part C: n = {nr:.0e}: device main terms {Q:.3e}, ratio to 3.69 n^(1/3) log2(n)^(2/3) = {Q / LF:.2f}")
res["genuine_cost_ratio"] = cost_ratio
print("RESULTS_JSON " + json.dumps(res))
