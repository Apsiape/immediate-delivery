"""Rounding flat rounds, and the twisted obstruction. A: the Diophantine constant behind Theorem G.12(a) and 2/(27 d ln 2) >= 0.0062
(d = 17). B: on twisted Folner squares, Lemma G.11, Lemma G.5 and Theorem G.12(a). C: the construction of Theorem G.9 on a leaky
Z-gadget round and on the thin rounds of Theorem G.12(b),(c). D: V_g (x) conj(V_g) removes scalar phase defects exactly, at doubled
bath entropy (remark in Appendix F.4). The last part also prints supplementary perturbed models.
Run: python -u check_rounding.py   (about 20 s, below 0.3 GB)
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadget_lib import build_gadget, word_op, z2_label, twisted_phase, kraus_from_gadget

res = {}
LN2 = np.log(2)
THETA = (np.sqrt(5) - 1) / 2


def S_bits(M, cut=1e-15):
    ev = np.linalg.eigvalsh((M + M.conj().T) / 2)
    ev = ev[ev > cut]
    return float(-(ev * np.log2(ev)).sum())


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def tn1(X):
    """normalized trace norm"""
    return float(np.linalg.svd(X, compute_uv=False).sum() / X.shape[0])


def dint(x):
    return abs(x - np.round(x))


# ------------------------------------------------------------------ gadgets: labels, Kraus operators, identities
def gadget(kind, omega=1.0 + 0j):
    """returns (G, labels, A, tri, one); tri = (gx, gy, gz, c) with the identity Y_(1,gy) = c Y_(gx,gz)."""
    if kind == 'Z':
        G = build_gadget(['a'], [])
        lab = lambda w: sum(1 if ch == 'a' else -1 for ch in w)
        ph = lambda w: 1.0 + 0j
        one = 0
    else:
        G = build_gadget(['a', 'b'], ['abAB'])
        lab = z2_label
        ph = lambda w: twisted_phase(w, omega)
        one = (0, 0)
    labels, A = kraus_from_gadget(G, lab, ph)
    tri = []
    for (x, y, z) in G['triangles']:
        zx, zy, zxy = ph(x), ph(y), ph(x + y)
        tri.append((lab(x), lab(y), lab(x + y), np.conj(zx) * zxy / zy))
    return G, labels, A, tri, one


def round_from_letters(G, letters, N):
    d = G['d']
    U = np.zeros((d * N, d * N), complex)
    for (row, col, c, w) in G['slots']:
        E = np.zeros((d, d))
        E[row, col] = 1
        U += c * np.kron(E, word_op(w, letters, N))
    return U


def project_classes(blocks, labels, tri, one):
    """orthogonal projection of the off-diagonal blocks onto {Y_(1,y) = c Y_(x,z) on triangles, Y_(h,g) = Y_(g,h)^*}."""
    nodes = [(g, h) for g in labels for h in labels if g != h]
    edges = {v: [] for v in nodes}
    for (g, h) in nodes:
        edges[(g, h)].append(((h, g), 1.0 + 0j, 1))
    for (gx, gy, gz, c) in tri:
        if gy == one:
            continue
        edges[(gx, gz)].append(((one, gy), c, 0))
        edges[(one, gy)].append(((gx, gz), 1 / c, 0))
    out, done = {}, set()
    for root in nodes:
        if root in done:
            continue
        rel = {root: (1.0 + 0j, 0)}
        stack, conflicts = [root], []
        while stack:
            u = stack.pop()
            cu, fu = rel[u]
            for (v, c, f) in edges[u]:
                cv, fv = c * (cu if f == 0 else np.conj(cu)), f ^ fu
                if v in rel:
                    if rel[v][1] != fv or abs(rel[v][0] - cv) > 1e-9:
                        conflicts.append((rel[v], (cv, fv)))
                else:
                    rel[v] = (cv, fv)
                    stack.append(v)
        done |= set(rel)
        tr = lambda X, f: X.conj().T if f else X
        R = sum(tr(blocks[v], f) / (c if f == 0 else np.conj(c)) for v, (c, f) in rel.items()) / len(rel)
        for ((c1, f1), (c2, f2)) in conflicts:
            if f1 == f2:
                R = 0 * R
            else:
                kap = c2 / c1 if f1 == 0 else np.conj(c1 / c2)
                R = (R + kap * R.conj().T) / 2
        for v, (c, f) in rel.items():
            out[v] = c * tr(R, f)
    return out


def herm_part(H, sign):
    ev, V = np.linalg.eigh((H + H.conj().T) / 2)
    return (V * np.clip(sign * ev, 0, None)) @ V.conj().T


def inv_sqrt(H):
    ev, V = np.linalg.eigh((H + H.conj().T) / 2)
    return (V / np.sqrt(ev)) @ V.conj().T


def theorem_F(U, G, labels, A, tri, one):
    """flat bath I/N on C^N; returns leakage, a, a', exactness, minimal eigenvalue of Y, m_B, the entropy line."""
    d, r = G['d'], len(labels)
    N = U.shape[0] // d
    ix = {g: i for i, g in enumerate(labels)}
    mu = {g: np.linalg.norm(A[g]) ** 2 / d for g in labels}
    Ub = U.reshape(d, N, d, N)
    X = {g: np.einsum('oq,oiqj->ij', A[g].conj(), Ub) / np.linalg.norm(A[g]) ** 2 for g in labels}
    L = U - sum(np.kron(A[g], X[g]) for g in labels)
    ell = np.linalg.norm(L) ** 2 / (d * N)
    Tst = (U @ np.kron(np.eye(d) / d, np.eye(N) / N) @ U.conj().T).reshape(d, N, d, N).trace(axis1=0, axis2=2)
    a = S_bits(Tst) - np.log2(N)
    sl = lambda g: slice(ix[g] * N, (ix[g] + 1) * N)
    Y0 = np.zeros((r * N, r * N), complex)
    for g in labels:
        for h in labels:
            Y0[sl(g), sl(h)] = X[g].conj().T @ X[h]
    P = sum(herm_part(X[g].conj().T @ X[g] - np.eye(N), +1) for g in labels)       # shrink
    IT = np.kron(np.eye(r), inv_sqrt(np.eye(N) + P))
    YA = IT @ Y0 @ IT
    proj = project_classes({(g, h): YA[sl(g), sl(h)] for g in labels for h in labels if g != h}, labels, tri, one)
    E = np.zeros_like(YA)
    for (g, h), Bk in proj.items():
        E[sl(g), sl(h)] = Bk - YA[sl(g), sl(h)]
    Em = herm_part(E, -1)
    Z = r * sum(Em[sl(g), sl(g)] for g in labels)                                   # absorbed negative part
    Delta = np.zeros_like(YA)
    for g in labels:
        Delta[sl(g), sl(g)] = np.eye(N) - YA[sl(g), sl(g)]
    YB = Delta + E + np.kron(np.eye(r), Z)
    Sc = np.kron(np.eye(r), inv_sqrt(np.eye(N) + Z))
    Y = Sc @ (YA + YB) @ Sc
    Y = (Y + Y.conj().T) / 2
    ev, V = np.linalg.eigh(Y)
    Yh = (V * np.sqrt(np.clip(ev, 0, None))) @ V.conj().T
    Xn = {g: Yh[:, sl(g)] for g in labels}
    iso = max(np.abs(Xn[g].conj().T @ Xn[g] - np.eye(N)).max() for g in labels)
    tri_err = max(np.abs(Xn[one].conj().T @ Xn[gy] - c * Xn[gx].conj().T @ Xn[gz]).max() for (gx, gy, gz, c) in tri)
    D = np.kron(np.diag([np.sqrt(mu[g]) for g in labels]), np.eye(N))
    a_new = S_bits(D @ Y @ D / N) - np.log2(N)
    mB = float(np.trace(D @ Sc @ YB @ Sc @ D).real / N)
    mBp = mB - ell
    line = a + ell * np.log2(N) + ell / LN2 + 2 * mBp * np.log2(d) + h2(mBp / (1 - ell)) + mB * np.log2(r / mB)
    gram = max(abs(np.trace(Y[sl(g), sl(h)])) / N for g in labels for h in labels if g != h)
    return dict(ell=ell, a=a, a_new=a_new, exact=max(iso, tri_err), minev=float(ev.min()), mB=mB, line=line, gram=gram)


# ------------------------------------------------------------------ Part A: Diophantine constant
ks = np.arange(1, 10 ** 6 + 1)
kd = ks * dint(ks * THETA)
res["dioph_min"] = float(kd.min())
res["dioph_argmin"] = int(ks[kd.argmin()])
res["dioph_min_k_ge_10"] = float(kd[9:].min())
res["obstruction_const"] = float(2 / (27 * 17 * LN2))
print(f"Part A: min_k<=1e6 k||k theta|| = {kd.min():.5f} at k = {ks[kd.argmin()]}; over k >= 10: {kd[9:].min():.5f}; "
      f"2/(27 d ln 2) = {res['obstruction_const']:.5f}")

# Part B: Lemma G.11 and Theorem G.12(a) on twisted Folner squares
omega = np.exp(2j * np.pi * THETA)
Gt, labt, At, trit, onet = gadget('T', omega)
d = Gt['d']
mut = {g: np.linalg.norm(At[g]) ** 2 / d for g in labt}
mumin = min(mut.values())
viol = dict(trace_norm_model=0, w=0, comm=0, a=0)
ratios = []
for n in [2, 3, 4, 6, 8, 10]:
    F = [(i, j) for i in range(n) for j in range(n)]
    Fp = list(F)
    for g in labt:
        for (i, j) in F:
            if (i + g[0], j + g[1]) not in Fp:
                Fp.append((i + g[0], j + g[1]))
    k, kp = len(F), len(Fp)
    B = {}
    for g in labt:
        Bm = np.zeros((kp, k), complex)
        for t, (i, j) in enumerate(F):
            Bm[Fp.index((i + g[0], j + g[1])), t] = omega ** (i * g[1])     # twisted regular representation
        B[g] = Bm
    tb = sum(mut[g] * B[g] @ B[g].conj().T for g in labt) / k
    a = S_bits(tb) - np.log2(k)
    ev, Q = np.linalg.eigh((tb + tb.conj().T) / 2)
    Qk = Q[:, ::-1][:, :k]
    w = 1 - float(ev[::-1][:k].sum())
    V = {}
    for g in labt:
        u_, s_, vh = np.linalg.svd(Qk.conj().T @ B[g])
        V[g] = u_ @ vh
    V = {g: V[onet].conj().T @ V[g] for g in labt}
    defect = max(tn1(V[gy] - c * V[gx].conj().T @ V[gz]) for (gx, gy, gz, c) in trit)
    Cm = V[(0, 1)] @ V[(1, 0)].conj().T @ V[(0, 1)].conj().T @ V[(1, 0)]
    comm = min(tn1(Cm - omega * np.eye(k)), tn1(Cm - np.conj(omega) * np.eye(k)))
    bound_a = 2 * dint(k * THETA) / (9 * d * LN2 * k)
    viol['trace_norm_model'] += defect > 6 * w / mumin + 1e-9
    viol['w'] += w > a * LN2 + 1e-12
    viol['comm'] += comm < 4 * dint(k * THETA) / k - 1e-9
    viol['a'] += a < bound_a - 1e-12
    ratios.append(a / bound_a)
    print(f"Part B: Folner n = {n:2d}, k = {k:3d}: a = {a:.4f}, w/(a ln 2) = {w / (a * LN2):.3f}, defect = {defect:.4f} "
          f"(6w/mu_min = {6 * w / mumin:.3f}), commutator = {comm:.4f} (>= {4 * dint(k * THETA) / k:.4f}), "
          f"a/bound = {a / bound_a:.1f}")
res["folner_violations"] = [int(viol[k_]) for k_ in ("trace_norm_model", "w", "comm", "a")]
res["folner_a_over_bound_min"] = float(min(ratios))

# ------------------------------------------------------------------ Part C: the rounding of Theorem G.9
Gz, labz, Az, triz, onez = gadget('Z')
rng = np.random.default_rng(5)                      # the leaky family of check_exact_region.py, Part C
N = 8
H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
Vz = expm(1j * (H + H.conj().T) / 2)
Kh = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
Kh = (Kh + Kh.conj().T) / 2
Kh /= np.linalg.norm(Kh, 2)
thetas = [0.3, 0.1, 0.03, 0.01, 0.003]
rowsZ = []
for th in thetas:
    U = round_from_letters(Gz, {'a': Vz, 'A': Vz.conj().T @ expm(1j * th * Kh)}, N)
    out = theorem_F(U, Gz, labz, Az, triz, onez)
    rowsZ.append(out)
    print(f"Part C(i): theta = {th:5.3f}: leakage {out['ell']:.3e}, a = {out['a']:.1e}, a' = {out['a_new']:.5f}, "
          f"line {out['line']:.4f}, exact to {out['exact']:.1e}, min eig Y {out['minev']:.1e}, Gram {out['gram']:.3f}")
res["F_Z_a_prime"] = [r_['a_new'] for r_ in rowsZ]
res["F_Z_a_before"] = [r_['a'] for r_ in rowsZ]
res["F_Z_exact"] = max(r_['exact'] for r_ in rowsZ)
res["F_Z_line_slack_min"] = min(r_['line'] - r_['a_new'] for r_ in rowsZ)
res["F_Z_slope"] = float(np.polyfit(np.log(thetas), np.log([r_['a_new'] for r_ in rowsZ]), 1)[0])
res["F_Z_over_sqrtl_log"] = [r_['a_new'] / (np.sqrt(r_['ell']) * np.log2(1 / r_['ell'])) for r_ in rowsZ]
print(f"  slope of a' in theta: {res['F_Z_slope']:.3f}")

rowsT = []
for Nf in [13, 21, 34, 55]:
    p = int(np.round(THETA * Nf))
    Sop = np.roll(np.eye(Nf), 1, axis=0).astype(complex)
    Om = np.diag(np.exp(2j * np.pi * p * np.arange(Nf) / Nf))
    U = round_from_letters(Gt, {'a': Sop, 'A': Sop.conj().T, 'b': Om, 'B': Om.conj().T}, Nf)
    out = theorem_F(U, Gt, labt, At, trit, onet)
    out['floor'] = 2 * dint(Nf * THETA) / (9 * d * LN2 * Nf)
    rowsT.append(out)
    print(f"Part C(ii): N = {Nf:2d}: leakage {out['ell']:.3e} (N^4 l = {out['ell'] * Nf ** 4:.3f}), a = {out['a']:.1e}, "
          f"a' = {out['a_new']:.4e} (a' N^2 = {out['a_new'] * Nf ** 2:.2f}; floor of Theorem G.12(a) at k = N: {out['floor']:.2e}), "
          f"line {out['line']:.3f}, exact to {out['exact']:.1e}")
res["F_thin_N4_leak"] = [r_['ell'] * Nf ** 4 for r_, Nf in zip(rowsT, [13, 21, 34, 55])]
res["F_thin_aN2"] = [r_['a_new'] * Nf ** 2 for r_, Nf in zip(rowsT, [13, 21, 34, 55])]
res["F_thin_exact"] = max(r_['exact'] for r_ in rowsT)
res["F_thin_line_slack_min"] = min(r_['line'] - r_['a_new'] for r_ in rowsT)
res["F_thin_above_floor"] = min(r_['a_new'] - r_['floor'] for r_ in rowsT)
res["F_thin_a_before"] = max(abs(r_['a']) for r_ in rowsT)

# ------------------------------------------------------------------ Part D: the adjoint family
Gu, labu, Au, triu, oneu = gadget('T', 1.0 + 0j)       # untwisted Z^2 gadget of abAB


def clockshift(Nn, p):
    Sop = np.roll(np.eye(Nn), 1, axis=0).astype(complex)
    Om = np.diag(np.exp(2j * np.pi * p * np.arange(Nn) / Nn))
    mp = np.linalg.matrix_power
    return {(u, v): mp(Sop, u % Nn) @ (mp(Om, v) if v >= 0 else mp(Om.conj().T, -v)) for (u, v) in labu}


def adj_defect_formula(R):
    """|| I - R (x) conj(R) ||_1 (normalized) for a unitary R, from its eigenvalues (R is normal)."""
    lam = np.linalg.eigvals(R)
    return float(np.abs(1 - np.outer(lam, lam.conj())).mean())


def min_phase_dist(R):
    """min over |c| = 1 of ||R - c I||_1 (normalized) for a unitary R: a grid, then a bounded local minimization."""
    from scipy.optimize import minimize_scalar
    lam = np.linalg.eigvals(R)
    f = lambda ph: float(np.abs(lam - np.exp(1j * ph)).mean())
    phis = np.linspace(0, 2 * np.pi, 4001)
    vals = [f(ph) for ph in phis]
    i = int(np.argmin(vals))
    o = minimize_scalar(f, bounds=(phis[max(i - 1, 0)], phis[min(i + 1, 4000)]), method='bounded',
                        options={'xatol': 1e-12})
    return float(min(o.fun, vals[i]))


dV, dW, sepV, sepW, aW, aF, formula_err = [], [], [], [], [], [], 0.0
for Nn in [16, 32, 64]:
    V = clockshift(Nn, 1)
    Rs = [V[gy].conj().T @ V[gx].conj().T @ V[gz] for (gx, gy, gz, c) in triu]
    dV.append(max(tn1(V[gy] - V[gx].conj().T @ V[gz]) for (gx, gy, gz, c) in triu))
    dW.append(max(adj_defect_formula(R) for R in Rs))
    if Nn == 16:                                         # direct computation on C^N (x) C^N
        W = {g: np.kron(V[g], V[g].conj()) for g in labu}
        direct = max(tn1(W[gy] - W[gx].conj().T @ W[gz]) for (gx, gy, gz, c) in triu)
        formula_err = abs(direct - dW[-1])
    sepV.append(max(abs(np.trace(V[g].conj().T @ V[h])) / Nn for g in labu for h in labu if g != h))
    sepW.append(max(abs(np.trace(V[g].conj().T @ V[h]) / Nn) ** 2 for g in labu for h in labu if g != h))
    if Nn <= 32:
        W = {g: np.kron(V[g], V[g].conj()) for g in labu}
        muu = {g: np.linalg.norm(Au[g]) ** 2 / Gu['d'] for g in labu}
        aW.append(S_bits(sum(muu[g] * W[g] @ W[g].conj().T for g in labu) / Nn ** 2) - 2 * np.log2(Nn))
    U = round_from_letters(Gu, {'a': V[(1, 0)], 'A': V[(-1, 0)], 'b': V[(0, 1)], 'B': V[(0, -1)]}, Nn)
    out = theorem_F(U, Gu, labu, Au, triu, oneu)
    aF.append(out['a_new'])
    print(f"Part D: clock/shift N = {Nn}: L1 defect of V {dV[-1]:.4f} (2 sin(pi/N) = {2 * np.sin(np.pi / Nn):.4f}), "
          f"of V(x)conj(V) {dW[-1]:.1e}; separations {sepV[-1]:.1e} -> {sepW[-1]:.1e}; leaky round of V: leakage "
          f"{out['ell']:.3e}, Theorem G.9 a' = {out['a_new']:.4f} at S' = log N")
res["adj_dV"] = dV
res["adj_dW"] = max(dW)
res["adj_formula_err"] = float(formula_err)
res["adj_aW"] = [float(abs(x)) for x in aW]
res["adj_rounding_a"] = aF
res["adj_sep"] = [max(sepV), max(sepW)]
# perturbed (non-scalar) models: W defect <= 2 min_c ||R - c||_1 and separation exactly squared
rng = np.random.default_rng(17)
viol_def, viol_sep, ratios = 0, 0, []
for trial in range(24):
    Nn = int(rng.choice([6, 8, 10, 12]))
    p = int(rng.integers(1, Nn))
    V = clockshift(Nn, p)
    Hh = rng.standard_normal((Nn, Nn)) + 1j * rng.standard_normal((Nn, Nn))
    pert = expm(1j * float(rng.choice([0.02, 0.05, 0.1, 0.3])) * (Hh + Hh.conj().T) / 2)
    letters = {'a': V[(1, 0)] @ pert, 'b': V[(0, 1)]}
    letters['A'], letters['B'] = letters['a'].conj().T, letters['b'].conj().T
    Vm = {g: None for g in labu}
    for g in labu:                                       # a unitary model: the word of a fixed representative
        u, v = g
        w = ('a' * u if u >= 0 else 'A' * (-u)) + ('b' * v if v >= 0 else 'B' * (-v))
        Vm[g] = word_op(w, letters, Nn)
    W = {g: np.kron(Vm[g], Vm[g].conj()) for g in labu}
    for (gx, gy, gz, c) in triu:
        R = Vm[gy].conj().T @ Vm[gx].conj().T @ Vm[gz]
        lhs = tn1(W[gy] - W[gx].conj().T @ W[gz])
        rhs = 2 * min_phase_dist(R)
        viol_def += lhs > rhs + 1e-9
        if rhs > 1e-9:
            ratios.append(lhs / rhs)
    for g in labu:
        for h in labu:
            if g != h:
                tv = np.trace(Vm[g].conj().T @ Vm[h]) / Nn
                tw = np.trace(W[g].conj().T @ W[h]) / Nn ** 2
                viol_sep += abs(tw - abs(tv) ** 2) > 1e-12
res["adj_pert_violations"] = [int(viol_def), int(viol_sep)]
res["adj_pert_ratio_max"] = float(max(ratios))
print(f"Part D: perturbed models: violations of the defect bound {viol_def}, of separation squared {viol_sep}; "
      f"max defect/bound {max(ratios):.3f}")
print("RESULTS_JSON " + json.dumps(res))
