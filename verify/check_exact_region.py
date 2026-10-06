"""Restriction to an exact region, Lemma G.7: on 1800 random flat rounds and subspaces of codimension nu k (nu = 1/8, 1/4, 1/2),
a_0 <= (a + nu log2(1/nu))/(1 - nu) + log2(1/(1 - nu)). Part A prints the path-register rate of Theorem G.4 on the spread-angle
round of Theorem I.1(a); Part C prints a supplementary rounding experiment the paper does not use.
Run: python -u check_exact_region.py   (a few seconds)
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm, sqrtm

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadget_lib import build_gadget, z2_label, kraus_from_gadget

res = {}


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def S_bits(r, cut=1e-13):
    ev = np.linalg.eigvalsh((r + r.conj().T) / 2)
    ev = ev[ev > cut]
    return float(-(ev * np.log2(ev)).sum())


# ------------------------------------------------------------------ Part A: path weights on the spread-angle family
Gd = build_gadget(['a'], [])
labs, A = kraus_from_gadget(Gd, z2_label, lambda w: 1.0)
d = Gd['d']
order = [(0, 0), (1, 0), (-1, 0)]
Ak = [A[g] for g in order]
mu = np.array([np.trace(a.conj().T @ a).real / d for a in Ak])
k = 6
U = np.roll(np.eye(k), 1, axis=0)
rate_ratio, rate_over_a, diag_err = [], [], 0.0
for th in [0.3, 0.1, 0.03, 0.01, 0.003]:
    c, s = np.cos(th), np.sin(th)
    B = [np.vstack([np.eye(k), np.zeros((k, k))]), np.vstack([c * U, s * U]), np.vstack([c * U.T, -s * U.T])]
    tau = sum(mu[g] * B[g] @ B[g].conj().T for g in range(3)) / k
    a = S_bits(tau) - np.log2(k)
    ev, vec = np.linalg.eigh(tau)
    E1, E2 = vec[:, np.argsort(ev)[::-1][:k]], vec[:, np.argsort(ev)[::-1][k:]]
    Cb = [[E.conj().T @ B[g] for g in range(3)] for E in (E1, E2)]
    M = [np.array([[np.trace(Cb[b][g].conj().T @ Cb[b][h]) / k for h in range(3)] for g in range(3)]) for b in range(2)]
    diag_err = max(diag_err, max(np.abs(M[b] - np.diag(np.diag(M[b]))).max() for b in range(2)))
    pi = np.array([[M[b][g, g].real for b in range(2)] for g in range(3)])
    hstar = max(h2(pi[g, 1]) for g in range(3))
    ustar = np.log2(sum(max(pi[:, b]) for b in range(2)))
    w = [np.linalg.eigvalsh(sum(M[b][h, g] * Ak[h].conj().T @ Ak[g] for g in range(3) for h in range(3))).max()
         for b in range(2)]
    beta = w[1]
    tc = 1 / np.log2(1 / beta)
    kap = max((1 - q) * w[0] ** (-tc) + q * w[1] ** (-tc) for q in np.linspace(0, min(1, w[1]), 2001))
    rate = np.log2(w[0] + w[1]) + np.log2(kap) / tc
    rate_ratio.append(float(rate / (hstar + ustar)))
    rate_over_a.append(float(rate / a))
res["paths_rate_over_spread_rate"] = rate_ratio
res["paths_rate_over_a"] = rate_over_a
res["paths_M_offdiag"] = float(diag_err)
print("Part A: rate / (h* + u*) =", [round(x, 2) for x in rate_ratio], "; rate / a =", [round(x, 2) for x in rate_over_a],
      f"; M^(b) diagonal to {diag_err:.1e}")

# ------------------------------------------------------------------ Part B: restriction to an exact region (Lemma G.7)
rng = np.random.default_rng(31)


def haar_iso(n, m):
    X = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    Q, R = np.linalg.qr(X)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def Tmap(W, dd, kk, M, P):
    sig = P / np.trace(P).real
    big = W @ np.kron(np.eye(dd) / dd, sig) @ W.conj().T
    return big.reshape(dd, M, dd, M).trace(axis1=0, axis2=2)


viol, ratios = 0, []
for trial in range(600):
    dd = 3; kk = int(rng.choice([8, 12, 16])); M = int(rng.choice([2, 3])) * kk
    W = haar_iso(dd * M, dd * kk)
    if trial % 2 == 0:
        Bs = [haar_iso(M, kk) for _ in range(dd)]
        W = np.zeros((dd * M, dd * kk), complex)
        for o in range(dd):
            W[o * M:(o + 1) * M, o * kk:(o + 1) * kk] = Bs[o] if o == 0 else Bs[0] @ (haar_iso(kk, kk) if rng.random() < 0.5 else np.eye(kk))
    a = S_bits(Tmap(W, dd, kk, M, np.eye(kk))) - np.log2(kk)
    for nu in [1 / 8, 1 / 4, 1 / 2]:
        kex = int(round((1 - nu) * kk)); nu_eff = 1 - kex / kk
        if trial % 3 == 0:
            V = haar_iso(kk, kex)
        else:
            H = rng.standard_normal((kk, kk)) + 1j * rng.standard_normal((kk, kk)); H = H + H.conj().T
            ev, vec = np.linalg.eigh(H)
            V = vec[:, :kex] if trial % 3 == 1 else vec[:, -kex:]
        a_ex = S_bits(Tmap(W, dd, kk, M, V @ V.conj().T)) - np.log2(kex)
        bound = (a + nu_eff * np.log2(1 / nu_eff)) / (1 - nu_eff) + np.log2(1 / (1 - nu_eff))
        viol += a_ex > bound + 1e-9
        if bound - a > 1e-12:
            ratios.append((a_ex - a) / (bound - a))
res["restrict_violations"] = int(viol)
res["restrict_trials"] = 1800
res["restrict_ratio_median_max"] = [float(np.median(ratios)), float(np.max(ratios))]
print(f"Part B: violations of Lemma G.7 {viol}/1800; (a_0 - a)/(bound - a): median {np.median(ratios):.3f}, max {np.max(ratios):.3f}")

# ------------------------------------------------------------------ Part C: a supplementary rounding experiment on a leaky round
rng = np.random.default_rng(5)
slots = Gd['slots']
lab = lambda w: sum(1 if ch == 'a' else -1 for ch in w)
labels = [0, 1, -1]
Az = {g: np.zeros((d, d), complex) for g in labels}
for (row, col, cf, w) in slots:
    Az[lab(w)][row, col] += cf
N = 8
H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)); V = expm(1j * (H + H.conj().T) / 2)
Kh = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)); Kh = (Kh + Kh.conj().T) / 2
Kh /= np.linalg.norm(Kh, 2)
muz = {g: np.linalg.norm(Az[g]) ** 2 / d for g in labels}
idx = {0: 0, 1: 1, -1: 2}


def l2_round(theta):
    letter = {'a': V, 'A': V.conj().T @ expm(1j * theta * Kh)}
    def word(w):
        Mw = np.eye(N, dtype=complex)
        for ch in w:
            Mw = Mw @ letter[ch]
        return Mw
    Uf = np.zeros((d * N, d * N), complex)
    for (row, col, cf, w) in slots:
        E = np.zeros((d, d)); E[row, col] = 1
        Uf += cf * np.kron(E, word(w))
    Ub = Uf.reshape(d, N, d, N)
    X = {g: np.einsum('oq,oiqj->ij', Az[g].conj(), Ub) / np.linalg.norm(Az[g]) ** 2 for g in labels}
    Lk = Uf - sum(np.kron(Az[g], X[g]) for g in labels)
    ell = np.linalg.norm(Lk) ** 2 / (d * N)
    rho = np.eye(N) / N
    T = (Uf @ np.kron(np.eye(d) / d, rho) @ Uf.conj().T).reshape(d, N, d, N).trace(axis1=0, axis2=2)
    a = S_bits(T, 1e-14) - np.log2(N)
    blk = lambda Y, g, h: Y[idx[g] * N:(idx[g] + 1) * N, idx[h] * N:(idx[h] + 1) * N]
    Y0 = np.zeros((3 * N, 3 * N), complex)
    for g in labels:
        for h in labels:
            Y0[idx[g] * N:(idx[g] + 1) * N, idx[h] * N:(idx[h] + 1) * N] = X[g].conj().T @ X[h]
    Yh = Y0.copy()
    def setb(Y, g, h, Bm):
        Y[idx[g] * N:(idx[g] + 1) * N, idx[h] * N:(idx[h] + 1) * N] = Bm
    for g in labels:
        setb(Yh, g, g, np.eye(N))
    avg = (blk(Y0, 0, -1) + blk(Y0, 1, 0)) / 2           # the class {(1, x^-1), (x, 1)} of the triangle (x, x^-1, 1)
    setb(Yh, 0, -1, avg); setb(Yh, 1, 0, avg); setb(Yh, -1, 0, avg.conj().T); setb(Yh, 0, 1, avg.conj().T)
    sh = max(0.0, -np.linalg.eigvalsh(Yh).min())
    Y = (Yh + sh * np.eye(3 * N)) / (1 + sh)
    Yhalf = sqrtm(Y)
    Xn = {g: Yhalf[:, idx[g] * N:(idx[g] + 1) * N] for g in labels}
    resid = max(np.linalg.norm(Xn[g].conj().T @ Xn[g] - np.eye(N), 2) for g in labels)
    resid = max(resid, np.linalg.norm(Xn[0].conj().T @ Xn[-1] - Xn[1].conj().T @ Xn[0], 2))
    gram = max(abs(np.trace(Xn[g].conj().T @ Xn[h])) / N for g in labels for h in labels if g != h)
    a_new = S_bits(sum(muz[g] * Xn[g] @ rho @ Xn[g].conj().T for g in labels), 1e-14) - np.log2(N)
    Dm = np.diag(np.repeat([np.sqrt(muz[g]) for g in labels], N))
    Yt0, Yt = Dm @ Y0 @ Dm, Dm @ Y @ Dm
    ell0 = 1 - np.trace(Yt0).real / N
    TF = 0.5 * np.abs(np.linalg.eigvalsh(Yt / N - Yt0 / (N * (1 - ell0)))).sum()
    bound = (a + ell0 * np.log2(N)) / (1 - ell0) + TF * np.log2(3 * N) + h2(TF)
    return ell, a, a_new, resid, gram, bound


thetas = [0.3, 0.1, 0.03, 0.01, 0.003]
rows = [l2_round(t) for t in thetas]
lt = np.log(thetas)
res["leaky_leak_over_theta2"] = [r[0] / t ** 2 for r, t in zip(rows, thetas)]
res["leaky_a_before"] = [r[1] for r in rows]
res["leaky_a_after"] = [r[2] for r in rows]
res["leaky_exact_resid"] = max(r[3] for r in rows)
res["leaky_gram_max"] = max(r[4] for r in rows)
res["leaky_bound_slack_min"] = min(r[5] - r[2] for r in rows)
res["leaky_slopes"] = [float(np.polyfit(lt, np.log([r[0] for r in rows]), 1)[0]),
                    float(np.polyfit(lt, np.log([abs(r[2] - r[1]) for r in rows]), 1)[0])]
for t, r in zip(thetas, rows):
    print(f"Part C: theta = {t:5.3f}: leakage {r[0]:.3e}, a = {r[1]:.1e}, a after rounding {r[2]:.4f}, "
          f"residual {r[3]:.1e}, Gram {r[4]:.3f}, bound {r[5]:.4f}")
print(f"  slopes in theta: leakage {res['leaky_slopes'][0]:.3f}, |a' - a| {res['leaky_slopes'][1]:.3f}")
print("RESULTS_JSON " + json.dumps(res))
