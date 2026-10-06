"""Houghton's channel: genuine rounds. A: Phi_H has d = 104 and 26 labels (Section 2.2). B: the exact local sector bound
(Lemma 8.3) holds with equality on flat standard representations of S_3^N and on random exact representations. C: the constants
of Theorems N.1 and N.2 (0.37891, 2 sqrt(104 ln 2) = 16.981, 2717, 0.0097, 385, 420) and the cap 7.13e-6 of Proposition I.4.
Later parts print supplementary quantities the paper does not use.
Run: python -u check_genuine.py   (about 10 s)
"""
import json
import math
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corner_tools import gadget as gd

LN2 = math.log(2)
rng = np.random.default_rng(20261001)
res = {}


def h2(t):
    return 0.0 if t <= 0 or t >= 1 else -t * math.log2(t) - (1 - t) * math.log2(1 - t)


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def vn(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


# ------------------------------------------------------------------ Part A: labels and the step law
mass = {}
for e in gd.ENT:
    g = gd.label(e[2])
    mass[g] = mass.get(g, 0.0) + 1.0 / e[4]
d = max(e[0] for e in gd.ENT) + 1
mu = {g: m / d for g, m in mass.items()}
step = {}                       # translation pi(g) -> total weight; the right regular action x -> x g^-1 steps by -pi(g)
for (t, h), m in mu.items():
    step[t] = step.get(t, 0.0) + m
drift = [sum(v * k[i] for k, v in step.items()) for i in (0, 1)]
res["d"], res["labels"] = d, len(mu)
res["min_mu_times_d"] = min(mu.values()) * d
res["step_law"] = [round(step[k], 5) for k in sorted(step)]
res["step_keys"] = [list(k) for k in sorted(step)]
res["drift"] = [round(x, 5) for x in drift]
res["max_abs_pi"] = [max(abs(k[0]) for k in step), max(abs(k[1]) for k in step)]
print(f"Part A: d = {d}, labels = {len(mu)}, min mu*d = {res['min_mu_times_d']:.3f}")
print("  translations pi(g) and weights:", {k: round(v, 5) for k, v in sorted(step.items())})
print(f"  label drift = ({drift[0]:.5f}, {drift[1]:.5f}); max |pi_1| = {res['max_abs_pi'][0]}, max |pi_2| = {res['max_abs_pi'][1]}")

# ------------------------------------------------------------------ Part B: exact local sector bound
w3 = np.exp(2j * np.pi / 3)
IRR = {'triv': (np.eye(1, dtype=complex), np.eye(1, dtype=complex)),
       'sign': (np.eye(1, dtype=complex), -np.eye(1, dtype=complex)),
       'std': (np.diag([w3, w3 ** 2]), np.array([[0, 1], [1, 0]], dtype=complex))}


def kron_list(ms):
    out = np.eye(1, dtype=complex)
    for m in ms:
        out = np.kron(out, m)
    return out


def rep(blocks, N):
    """direct sum over blocks of tensor products of irreps: copies (c_i, x_i), i < N, of an exact rep of S_3^N"""
    Cs, Xs = [], []
    for i in range(N):
        cb = [kron_list([IRR[b[k]][0] if k == i else np.eye(len(IRR[b[k]][0])) for k in range(N)]) for b in blocks]
        xb = [kron_list([IRR[b[k]][1] if k == i else np.eye(len(IRR[b[k]][0])) for k in range(N)]) for b in blocks]
        D = sum(len(c) for c in cb)
        C, X, p = np.zeros((D, D), complex), np.zeros((D, D), complex), 0
        for c, x in zip(cb, xb):
            k = len(c); C[p:p + k, p:p + k] = c; X[p:p + k, p:p + k] = x; p += k
        Cs.append(C); Xs.append(X)
    return Cs, Xs


def sqrtm_psd(r):
    w, V = np.linalg.eigh((r + r.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def elsb_rhs(Cs, Xs, rho):
    xi = sqrtm_psd(rho)
    tot = 0.0
    for C, X in zip(Cs, Xs):
        Dl = np.linalg.norm((C - np.eye(len(C))) @ xi)
        th = np.linalg.norm(C @ xi - xi @ C)
        et = np.linalg.norm(X @ xi - xi @ X)
        tot += Dl ** 2 / 3 - et ** 2 / (2 * LN2) - h2(th ** 2 / 3) - th ** 2 / 3
    return tot


flat = []
for N in range(1, 5):
    Cs, Xs = rep([tuple(['std'] * N)], N)
    rho = np.eye(2 ** N) / 2 ** N
    flat.append(abs(vn(rho) - elsb_rhs(Cs, Xs, rho)))
min_slack = np.inf
for trial in range(300):
    N = int(rng.integers(1, 4))
    blocks = [tuple(rng.choice(['triv', 'sign', 'std'], size=N, p=[0.15, 0.15, 0.7])) for _ in range(int(rng.integers(1, 5)))]
    Cs, Xs = rep(blocks, N)
    D = len(Cs[0])
    V = np.linalg.qr(rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D)))[0]
    Cs = [V @ C @ V.conj().T for C in Cs]; Xs = [V @ X @ V.conj().T for X in Xs]
    G = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    rand = G @ G.conj().T; rand /= np.trace(rand).real
    lam = rng.uniform(0, 1) ** 3
    cent = V @ np.diag(rng.dirichlet(np.ones(D))) @ V.conj().T
    rho = (1 - lam) * cent + lam * rand
    min_slack = min(min_slack, vn(rho) - elsb_rhs(Cs, Xs, rho))
res["elsb_flat_equality_err"] = max(flat)
res["elsb_min_slack"] = float(min_slack)
print(f"Part B: |S - RHS| on the flat standard rep, N=1..4: max {max(flat):.1e}; random exact reps: min slack {min_slack:.2e}")

# ------------------------------------------------------------------ Part C: constants
term = (math.sqrt(1.5) - 0.1) ** 2 / 3 - 0.01 / (2 * LN2) - h2(1 / 300) - 1 / 300
c_chi = 2 * math.sqrt(104 * LN2)
inv_A = 60 * 16.981 / 0.375
cost = (2 ** (1 / 3) + 2 ** (-2 / 3)) * (1 / 2717) ** (2 / 3)
a2_c1 = 2717 * math.sqrt(7.4)
a2_c = (a2_c1 * (1 + 0.25 / 14)) ** (2 / 3)
# A2': n a' <= 6.4Q + 13.52 + 0.134 log n + 2Q + 2 n h2(1/n); n h2(1/n) <= log2 n + log2 e for n >= 2
grid = np.unique(np.logspace(np.log10(2), 12, 400).astype(np.int64))
nh2 = lambda n: math.log2(n) + (n - 1) * math.log1p(1 / (n - 1)) / LN2      # n h2(1/n), computed stably
nh2_margin = max(nh2(n) - math.log2(n) - math.log2(math.e) for n in grid)
a2p_const = 13.52 + 2 * math.log2(math.e)
a2p_slope = 0.134 + 2
a2p_c1 = 2717 * math.sqrt(9.4)
a2p_c = (a2p_c1 * (1 + 0.25 / a2p_const)) ** (2 / 3)
g = lambda x: (x * math.log(x) - x + 1) / (math.sqrt(x) - 1) ** 2
xs = np.linspace(1e-6, 52, 200001)
gx = (xs * np.log(xs) - xs + 1) / (np.sqrt(xs) - 1) ** 2
g_monotone = bool(np.all(np.diff(gx[np.isfinite(gx)]) > -1e-9))
folner_const = g(52) / LN2 * 4 * math.pi ** 2 * 13
h7 = (16 / (9 * 1e8 * 2 ** 17)) ** 0.4
res.update(per_copy_term=term, c_chi=c_chi, inv_A=inv_A, cost_const=cost, a2_c1=a2_c1, a2_c=a2_c,
           nh2_margin=nh2_margin, a2p_const=a2p_const, a2p_slope=a2p_slope, a2p_c1=a2p_c1, a2p_c=a2p_c,
           g52=g(52), g_monotone=1.0 if g_monotone else 0.0, folner_const=folner_const, h7_cap=h7,
           min_mu_inverse=1 / min(step.values()))
print(f"Part C: per-copy term {term:.5f}; 2 sqrt(104 ln 2) = {c_chi:.4f}; 60*16.981/0.375 = {inv_A:.1f}; "
      f"cost constant {cost:.5f}")
print(f"  A2: 2717 sqrt(7.4) = {a2_c1:.1f}, Q >= n^(1/3)/{a2_c:.1f};  A2': constant {a2p_const:.2f} + {a2p_slope:.3f} log n, "
      f"2717 sqrt(9.4) = {a2p_c1:.1f}, Q >= n^(1/3)/{a2p_c:.1f}; max_n [n h2(1/n) - log n - log e] = {nh2_margin:.2e}")
print(f"  g increasing on (0, 52]: {g_monotone}, g(52) = {g(52):.4f}; Folner constant (g(52)/ln2) 4 pi^2 13 = {folner_const:.1f}; "
      f"1/min step weight = {1 / min(step.values()):.2f}; one-round cap N <= {h7:.3e} nu^(-1/5)")

# ------------------------------------------------------------------ Part D: exact Folner rounds
steps = list(step.items())


def a_of(q):
    Mp, pad = q.shape[0], 4
    big = np.zeros((Mp + 2 * pad, Mp + 2 * pad))
    for (u, v), w in steps:
        big[pad + u:pad + u + Mp, pad + v:pad + v + Mp] += w * q
    return entropy_bits(big) - entropy_bits(q)


def bump(Mp, k=1):
    x = np.arange(1, Mp + 1)
    s = np.sin(np.pi * x / (Mp + 1)) ** (2 * k)
    s /= s.sum()
    return np.outer(s, s)


aM = []
for Mp in [16, 32, 64, 128, 256, 512]:
    aM.append(a_of(bump(Mp)) * Mp ** 2)
cst = aM[-1]
ratios = []
for e10 in range(6, 19, 2):
    n = 10.0 ** e10
    best = min(math.lgamma(2 * M + 1) / LN2 + 2 * math.log2(M - 24) + n * cst / (M - 24) ** 2
               for M in np.unique(np.logspace(np.log10(28), 8, 4000).astype(int)))
    ratios.append(best / (n ** (1 / 3) * math.log2(n) ** (2 / 3)))
res["folner_aM2"] = [round(x, 2) for x in aM]
res["folner_aM2_change_256_512"] = abs(aM[-1] - aM[-2]) / aM[-2]
res["folner_cost_ratio_range"] = [min(ratios), max(ratios)]
res["folner_bound_slack"] = min(2965 / (Mp + 1) ** 2 / (x / Mp ** 2) for Mp, x in zip([16, 32, 64, 128, 256, 512], aM))
print("Part D: a M'^2 =", res["folner_aM2"], f"(change 256->512: {res['folner_aM2_change_256_512']:.4f}); "
      f"L_F/(n^(1/3) log2(n)^(2/3)) in [{min(ratios):.3f}, {max(ratios):.3f}] for n = 1e6..1e18; "
      f"proved bound / measured >= {res['folner_bound_slack']:.0f}")

# ------------------------------------------------------------------ Part E: sin^(2k) bumps
base = None
dev = []
for k in range(1, 7):
    v = a_of(bump(512, k)) * 512 ** 2
    base = base or v
    dev.append(abs(v / base / (k * k / (2 * k - 1)) - 1))
res["sin2k_max_rel_dev"] = max(dev)
print(f"Part E: max relative deviation of a(k)/a(1) from k^2/(2k-1), k = 1..6, M' = 512: {max(dev):.4f}")

# ------------------------------------------------------------------ Part F: biased corner clocks
rows = []
for M in [16, 24, 32, 48, 64, 96, 128]:
    n_ = M * M
    idx = np.arange(n_).reshape(M, M)
    ii, jj = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    di = np.minimum((ii - (M - 1)) % M, ((M - 1) - ii) % M)
    dj = np.minimum((jj - (M - 1)) % M, ((M - 1) - jj) % M)
    hole = np.maximum(di, dj) <= 3
    Lap = sp.csr_matrix((n_, n_))
    for (u, v), w in steps:
        if (u, v) == (0, 0):
            continue
        T = sp.csr_matrix((np.ones(n_), (idx.ravel(), np.roll(np.roll(idx, -u, 0), -v, 1).ravel())), shape=(n_, n_))
        Lap = Lap + w * (2 * sp.eye(n_) - T - T.T)
    keep = ~hole.ravel()
    vals, vecs = sla.eigsh(Lap[keep][:, keep].tocsc(), k=1, sigma=0, which="LM")
    phi = np.zeros(n_); phi[keep] = np.abs(vecs[:, 0])
    W = (phi ** 2 / (phi ** 2).sum()).reshape(M, M)
    U = (~hole).astype(float); U /= U.sum()
    bar, baru = np.zeros((M, M)), np.zeros((M, M))
    for (u, v), w in steps:
        bar += w * np.roll(np.roll(W, u, 0), v, 1)
        baru += w * np.roll(np.roll(U, u, 0), v, 1)
    rows.append((M, (entropy_bits(bar.ravel()) - entropy_bits(W.ravel())) * M * M,
                 (entropy_bits(baru.ravel()) - entropy_bits(U.ravel())) * M * M))
Ms = np.array([r[0] for r in rows], float)
aM2 = np.array([r[1] for r in rows])
slope, icpt = np.polyfit(np.log(Ms), 1 / aM2, 1)
res["corner_aM2"] = [round(float(x), 2) for x in aM2]
res["corner_flat_aM2_128"] = rows[-1][2]
res["corner_fit"] = [slope, icpt]
res["corner_fit_max_rel_resid"] = float(np.max(np.abs((slope * np.log(Ms) + icpt) * aM2 - 1)))
print("Part F: biased clocks a M^2 =", res["corner_aM2"], f"(M = 16..128); flat minus hole at M = 128: {rows[-1][2]:.2f}; "
      f"fit 1/(a M^2) = {slope:.3f} ln M {icpt:+.3f} (max rel. residual {res['corner_fit_max_rel_resid']:.3f})")

print("RESULTS_JSON " + json.dumps(res))
