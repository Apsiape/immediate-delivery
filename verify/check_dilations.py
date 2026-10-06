"""Exact rectangular dilations (Definition 2.8). A: tensor powers of a gadget family have entrywise-power flat Gram
(Theorem 7.6(b)). B: the Gram-completion step in the proof of Lemma 4.3 makes every label trace exact. C: a dilation gives an
exact round with entropy production at most log2(k'/k) (Lemma 2.9(c)). D: thin clock-and-shift rounds of the twisted gadget
have leakage of order e^2 (Lemma L.1, Theorem G.12(b)). Some parts also print supplementary quantities the paper does not use.
Run: python -u check_dilations.py   (about 40 s)
"""
import os
import sys
import json
import time
import numpy as np
import scipy.sparse as sps
from scipy.linalg import null_space

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadget_lib import build_gadget, word_op, z2_label, twisted_phase, kraus_from_gadget, choi_from_slot_gram, choi_from_kraus

results = {}
t00 = time.time()

# ---------------------------------------------------------------- Part A: rank bridge and tensor powers
print("=== Part A: rank bridge (random exact triangles) and tensor-power Gram ===")
rng = np.random.default_rng(1)


def rand_unitary(n):
    z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def extend_to_unitary(B):
    kp, k = B.shape
    Z = rng.normal(size=(kp, kp - k)) + 1j * rng.normal(size=(kp, kp - k))
    Z -= B @ (B.conj().T @ Z)
    Q, _ = np.linalg.qr(Z)
    return np.hstack([B, Q])


def rank(X, tol=1e-9):
    s = np.linalg.svd(X, compute_uv=False)
    return int((s > tol * max(1, s[0])).sum())


worst, tri_err = 0.0, 0.0
for trial in range(200):
    k = int(rng.integers(3, 12)); m = int(rng.integers(1, 4)); kp = k + m
    px = rand_unitary(kp); py = rand_unitary(kp)
    pz = extend_to_unitary((px @ py)[:, :k])
    iota = np.eye(kp)[:, :k]
    Bx, By, Bz = px[:, :k], py[:, :k], pz[:, :k]
    tri_err = max(tri_err, np.abs(iota.conj().T @ By - Bx.conj().T @ Bz).max())
    ex, ey, ez = extend_to_unitary(Bx), extend_to_unitary(By), extend_to_unitary(Bz)
    worst = max(worst, rank(ex @ ey - ez) / (2 * m))
print(f"200 random exact triangles (triangle identity error {tri_err:.1e}): "
      f"max rank(pi_x pi_y - pi_z)/(2(k'-k)) = {worst:.3f}")
results["bridge_max_ratio"] = worst


def shift_block(s):
    P = np.zeros((7, 7))
    for w in range(2):
        P[(w + s) % 2, w] = 1
    for w in range(5):
        P[2 + (w + s) % 5, 2 + w] = 1
    return P


B = {g: shift_block(g)[:, :7] for g in (0, 1, -1)}
G = np.array([[np.trace(B[g].T @ B[h]) / 7 for h in (0, 1, -1)] for g in (0, 1, -1)])
Bt = dict(B)
for _ in range(2):
    Bt = {g: np.kron(Bt[g], B[g]) for g in B}
tri3 = np.abs(Bt[0].T @ Bt[-1] - Bt[1].T @ Bt[0]).max()
Gt = np.array([[np.trace(Bt[g].T @ Bt[h]) / Bt[0].shape[1] for h in (0, 1, -1)] for g in (0, 1, -1)])
print(f"Z_2 (+) Z_5 shift model: G(x, x^-1) = {G[1, 2]:.4f}; third tensor power: triangle error {tri3:.1e}, "
      f"Gram entry {Gt[1, 2]:.5f} = G^3 entrywise: {np.allclose(Gt, G ** 3)}")
results["gram_z2z5"] = G[1, 2]
results["gram_z2z5_cubed"] = Gt[1, 2]
results["tensor_power_gram_is_power"] = float(np.allclose(Gt, G ** 3))

# ---------------------------------------------------------------- Part B: trace correction
print("\n=== Part B: trace correction ===")
rng = np.random.default_rng(7)


def family_stats(V, triangles, labels):
    k = V[labels[0]].shape[1]
    iso = max(np.abs(V[g].conj().T @ V[g] - np.eye(k)).max() for g in labels)
    tri = max(np.abs(V[x].conj().T @ V[z] - V[0].conj().T @ V[y]).max() for (x, y, z) in triangles)
    T = np.array([[np.trace(V[g].conj().T @ V[h]) for h in labels] for g in labels])
    off = np.abs(T - np.diag(np.diag(T))).max() / k
    return iso, tri, off


def correct(V, labels):
    """Gram completion: returns the blocks W_g = c_g (x) I_N of the direct sum V_g (+) W_g."""
    k = V[labels[0]].shape[1]
    T = np.array([[np.trace(V[g].conj().T @ V[h]) for h in labels] for g in labels])
    X = T - k * np.eye(len(labels))
    lam = np.linalg.eigvalsh((X + X.conj().T) / 2).max()
    N = max(1, int(np.ceil(lam - 1e-12)))
    Gt_ = np.eye(len(labels)) - X / N
    w, Q = np.linalg.eigh((Gt_ + Gt_.conj().T) / 2)
    assert w.min() > -1e-9
    keep = w > 1e-10
    C = (Q[:, keep] * np.sqrt(w[keep])).conj().T
    r = C.shape[0]
    Wb = {g: np.kron(C[:, i:i + 1], np.eye(N)) for i, g in enumerate(labels)}
    return Wb, N, r


def sum_stats(V, Wb, triangles, labels):
    """isometry, triangle and trace statistics of the direct sum V_g (+) W_g, computed block by block."""
    iso_v, tri_v, _ = family_stats(V, triangles, labels)
    iso_w, tri_w, _ = family_stats(Wb, triangles, labels)
    k = V[labels[0]].shape[1] + Wb[labels[0]].shape[1]
    T = np.array([[np.trace(V[g].conj().T @ V[h]) + np.trace(Wb[g].conj().T @ Wb[h]) for h in labels] for g in labels])
    off = np.abs(T - np.diag(np.diag(T))).max() / k
    diag = np.abs(np.diag(T) - k).max()
    return max(iso_v, iso_w), max(tri_v, tri_w), off, diag


labels = [0, 1, -1, 2]
triangles = [(1, 1, 2), (1, -1, 0), (-1, 1, 0)]
after_off, after_err = [], []
# test 1: exact representation of Z (square, large traces)
k = 40
phases = rng.uniform(0, 2 * np.pi, k) * 0.15
u = np.diag(np.exp(1j * phases))
V = {g: np.linalg.matrix_power(u, g) if g >= 0 else np.linalg.matrix_power(u.conj().T, -g) for g in labels}
print("Z representation, before: iso/tri/off = %.1e %.1e %.4f" % family_stats(V, triangles, labels))
Wb, N, r = correct(V, labels)
st = sum_stats(V, Wb, triangles, labels)
after_off.append(st[2]); after_err.append(max(st[0], st[1], st[3]))
print("Z representation, after:  iso/tri/off = %.1e %.1e %.4f" % st[:3])
# test 2: partial permutations with collisions
M = 300
px = rng.permutation(M); px_inv = np.argsort(px)
p2 = px[px].copy()
bad = rng.choice(M, 6, replace=False)
p2[bad] = rng.permutation(p2[bad])
f = {0: np.arange(M), 1: px, -1: px_inv, 2: p2}
good = [w for w in range(M) if all(f[x][f[y]][w] == f[z][w] for (x, y, z) in triangles)]
V2 = {}
for g in labels:
    Bm = np.zeros((M, len(good)))
    for j, w in enumerate(good):
        Bm[f[g][w], j] = 1.0
    V2[g] = Bm
print("partial permutations, before: iso/tri/off = %.1e %.1e %.4f" % family_stats(V2, triangles, labels))
Wb2, N2, r2 = correct(V2, labels)
st = sum_stats(V2, Wb2, triangles, labels)
after_off.append(st[2]); after_err.append(max(st[0], st[1], st[3]))
print("partial permutations, after:  iso/tri/off = %.1e %.1e %.4f" % st[:3])
# test 3: tensor powers, then correction
k = 12
phases = np.linspace(0, 2 * np.pi, k, endpoint=False) * 0.55 + 0.3
u = np.diag(np.exp(1j * phases))
base = {g: np.linalg.matrix_power(u, g) if g >= 0 else np.linalg.matrix_power(u.conj().T, -g) for g in labels}
rooms, befores = [], []
for t in [1, 2, 3]:
    Vt = {g: base[g] for g in labels}
    for _ in range(t - 1):
        Vt = {g: np.kron(Vt[g], base[g]) for g in labels}
    st0 = family_stats(Vt, triangles, labels)
    Wb3, N3, r3 = correct(Vt, labels)
    st = sum_stats(Vt, Wb3, triangles, labels)
    after_off.append(st[2]); after_err.append(max(st[0], st[1], st[3]))
    del Wb3
    kt = Vt[0].shape[1]
    room = np.log2((kt + r3 * N3) / (kt + N3))
    rooms.append(room); befores.append(st0[2])
    print(f"t={t}: max label trace/k before {st0[2]:.4f}, after {st[2]:.1e}; room log2(k'/k) = {room:.4f}")
results["tc_after_max_off"] = max(after_off)
results["tc_after_max_err"] = max(after_err)
results["tc_rooms"] = rooms
results["tc_before"] = befores

# ---------------------------------------------------------------- Part C: Phi_p, and E <= R
print("\n=== Part C: Phi_p (no flat probe) and the comparison E(0, d) <= R(d) ===")


def choi(kraus, d):
    J = np.zeros((d * d, d * d), complex)
    for K in kraus:
        v = np.zeros(d * d, complex)
        for i in range(d):
            e = np.zeros(d); e[i] = 1
            v += np.kron(K @ e, e)
        J += np.outer(v, v.conj())
    return J / d


def channel_of_dilation(W, d, k, kp):
    J = np.zeros((d * d, d * d), complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d)); Eij[i, j] = 1
            out = W @ np.kron(Eij, np.eye(k) / k) @ W.conj().T
            out = out.reshape(d, kp, d, kp).trace(axis1=1, axis2=3)
            J += np.kron(out, Eij) / d
    return J


def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-15]
    return float(-(w * np.log2(w)).sum())


def round_from_dilation(W, d, k, kp, kraus):
    emb = np.zeros((kp, k)); emb[:k, :k] = np.eye(k)
    Ein = np.kron(np.eye(d), emb)
    U = W @ Ein.conj().T + null_space(W.conj().T) @ null_space(Ein.conj().T).conj().T
    assert np.allclose(U.conj().T @ U, np.eye(d * kp), atol=1e-10)
    rho = emb @ emb.T / k
    J = np.zeros((d * d, d * d), complex)
    for i in range(d):
        for j in range(d):
            Eij = np.zeros((d, d)); Eij[i, j] = 1
            out = U @ np.kron(Eij, rho) @ U.conj().T
            J += np.kron(out.reshape(d, kp, d, kp).trace(axis1=1, axis2=3), Eij) / d
    Jphi = choi(kraus, d)
    w, Vv = np.linalg.eigh(Jphi)
    Pi = Vv[:, w > 1e-10] @ Vv[:, w > 1e-10].conj().T
    leak = float(np.real(np.trace((np.eye(d * d) - Pi) @ J)))
    T = (U @ np.kron(np.eye(d) / d, rho) @ U.conj().T).reshape(d, kp, d, kp).trace(axis1=0, axis2=2)
    return np.abs(J - Jphi).max(), leak, entropy(T) - entropy(rho), np.log2(kp / k)


p = (np.sqrt(5) - 1) / 2
D = np.diag([1, 1j, -1, -1j])
A1, A2 = np.sqrt(p) * np.eye(4), np.sqrt(1 - p) * D
rkID = np.linalg.matrix_rank(np.array([np.eye(4).ravel(), D.ravel(), D.conj().ravel()]))
XY = np.linalg.solve(np.array([[p, 1 - p], [1.0, 1.0]]), np.array([1.0, 2.0]))
print(f"rank of {{I, D, D*}} = {rkID}; flat probe would need Tr(rho A1*A1) = 1/2, but it is p = {p:.4f}")
print(f"pX + (1-p)Y = 1 and X + Y = 2 give (X, Y) = ({XY[0]:.6f}, {XY[1]:.6f}): orthogonal isometries, k' >= 2k")
results["phip_rank_IDD"] = rkID
results["phip_XY"] = [float(XY[0]), float(XY[1])]
fails = []
for k in [3, 5, 8, 13]:
    n1 = int(np.floor(p * k)); n2 = k - 1 - n1
    xA = (p * k - n1) / p; yA = (1 - p * xA) / (1 - p)
    x = [1 / p] * n1 + [0.0] * n2 + [xA]
    y = [0.0] * n1 + [1 / (1 - p)] * n2 + [yA]
    kp = sum(1 for t in x if t > 0) + sum(1 for t in y if t > 0)
    B1 = np.zeros((kp, k)); B2 = np.zeros((kp, k)); c = 0
    for j in range(k):
        if x[j] > 0:
            B1[c, j] = np.sqrt(x[j]); c += 1
    for j in range(k):
        if y[j] > 0:
            B2[c, j] = np.sqrt(y[j]); c += 1
    W = np.kron(A1, B1) + np.kron(A2, B2)
    iso = np.abs(W.conj().T @ W - np.eye(4 * k)).max()
    trc = np.abs(np.array([[np.trace(Bi.T @ Bj) for Bj in (B1, B2)] for Bi in (B1, B2)]) - k * np.eye(2)).max()
    jd = np.abs(channel_of_dilation(W, 4, k, kp) - choi([A1, A2], 4)).max()
    second = np.abs(B1.T @ B1 + B2.T @ B2 - 2 * np.eye(k)).max()
    dist, leak, a, room = round_from_dilation(W, 4, k, kp, [A1, A2])
    fails.append(second)
    print(f"k={k:2d} k'={kp:2d}: |W*W-I|={iso:.1e} |Tr(Bi*Bj)-k delta|={trc:.1e} |J_dil-J_Phi|={jd:.1e} "
          f"|sum Bi*Bi-2I|={second:.3f}; round: dist={dist:.1e} leak={leak:.1e} a={a:.4f} <= {room:.4f}")
    results.setdefault("phip_first_only_errs", []).append(max(iso, trc, jd, dist, abs(leak)))
    results.setdefault("phip_first_only_kprime_minus_k", []).append(kp - k)
    results.setdefault("phip_a_minus_room", []).append(a - room)
results["phip_second_identity_violation"] = min(fails)
# E <= R on a flat-probe channel (equal mixture of 4 Weyl unitaries on C^3)
w3 = np.exp(2j * np.pi / 3)
Xs = np.roll(np.eye(3), 1, axis=0); Zs = np.diag([1, w3, w3 ** 2])
Us = [np.linalg.matrix_power(Xs, a_) @ np.linalg.matrix_power(Zs, b_) for a_ in range(3) for b_ in range(3)][:4]
kraus = [U / 2 for U in Us]
for (k, kp) in [(4, 4), (4, 6)]:
    Bs = []
    for i in range(4):
        Bm = np.zeros((kp, k)); Bm[i, i] = 2.0; Bs.append(Bm)
    W = sum(np.kron(A, Bm) for A, Bm in zip(kraus, Bs))
    dist, leak, a, room = round_from_dilation(W, 3, k, kp, kraus)
    print(f"Weyl channel, k={k} k'={kp}: round dist={dist:.1e} leak={leak:.1e} a={a:.4f} <= log(k'/k)={room:.4f}")
    results.setdefault("weyl_round_errs", []).append(max(dist, abs(leak)))
    results.setdefault("weyl_a_minus_room", []).append(a - room)

# ---------------------------------------------------------------- Part D: twisted Z^2 test bed
print("\n=== Part D: the twisted-Z^2 gadget ===")
theta = (np.sqrt(5) - 1) / 2
Gd = build_gadget(['a', 'b'], ['abAB'])
d = Gd['d']


def setup(omega):
    return kraus_from_gadget(Gd, z2_label, lambda w: twisted_phase(w, omega))


def triangle_data(omega):
    return [(z2_label(x), z2_label(y), z2_label(x + y), twisted_phase(x, omega), twisted_phase(y, omega),
             twisted_phase(x + y, omega)) for (x, y, z) in Gd['triangles']]


def check_dilation(Bs, labels, A, omega, k):
    W = None
    for g in labels:
        term = sps.kron(sps.csr_matrix(A[g]), Bs[g])
        W = term if W is None else W + term
    WW = (W.conj().T @ W).tocsr() - sps.identity(d * k, format='csr')
    iso = abs(WW).max() if WW.nnz else 0.0
    gram = max(abs((Bs[g].conj().T @ Bs[h]).diagonal().sum() - (k if g == h else 0)) for g in labels for h in labels)
    tri = 0.0
    for (gx, gy, gz, zx, zy, zxy) in triangle_data(omega):
        Dm = zy * (Bs[(0, 0)].conj().T @ Bs[gy]) - np.conj(zx) * zxy * (Bs[gx].conj().T @ Bs[gz])
        if Dm.nnz:
            tri = max(tri, abs(Dm).max())
    return iso, gram, tri


def rank_bridge(Bs, labels, omega, k, kp):
    B1 = Bs[(0, 0)].toarray()
    rows_in = np.where(np.abs(B1).sum(axis=1) > 0.5)[0]
    order = np.argmax(np.abs(B1[rows_in, :]), axis=1)
    perm_rows = rows_in[np.argsort(order)]
    Ups = {}
    for g in labels:
        Cm = Bs[g].toarray()[perm_rows, :]
        Uu, s, Vh = np.linalg.svd(Cm)
        Ups[g] = Uu @ Vh
    wr = 0
    for (gx, gy, gz, zx, zy, zxy) in triangle_data(omega):
        Dm = zy * Ups[gy] - np.conj(zx) * zxy * Ups[gx].conj().T @ Ups[gz]
        wr = max(wr, int((np.linalg.svd(Dm, compute_uv=False) > 1e-9).sum()))
    wt = max(abs(np.trace(Ups[g].conj().T @ Ups[h])) for g in labels for h in labels if g != h)
    return kp - k, wr, wt


folner_ok = True
for name, omega in [("Z^2", 1.0 + 0j), ("twisted Z^2", np.exp(2j * np.pi * theta))]:
    labels, A = setup(omega)
    for n in [2, 3, 4, 6, 8, 10]:
        F = [(i, j) for i in range(n) for j in range(n)]
        Fp = list(F)
        for g in labels:
            for (i, j) in F:
                q = (i + g[0], j + g[1])
                if q not in Fp:
                    Fp.append(q)
        idx = {f_: t for t, f_ in enumerate(F)}
        idxp = {f_: t for t, f_ in enumerate(Fp)}
        k, kp = len(F), len(Fp)
        Bs = {}
        for g in labels:
            u_, v_ = g
            rows, cols, vals = [], [], []
            for (i, j) in F:
                rows.append(idxp[(i + u_, j + v_)]); cols.append(idx[(i, j)]); vals.append(omega ** (i * v_))
            Bs[g] = sps.csr_matrix((vals, (rows, cols)), shape=(kp, k))
        iso, gram, tri = check_dilation(Bs, labels, A, omega, k)
        m, wr, wt = rank_bridge(Bs, labels, omega, k, kp)
        ok = (kp == n * n + 4 * n + 1 and iso < 1e-10 and gram < 1e-9 and tri < 1e-10 and wr <= 4 * m and wt <= 3 * m + 1e-9)
        folner_ok = folner_ok and ok
    print(f"[{name}] Folner squares n = 2..10: exact, k' = n^2 + 4n + 1, bridge bounds hold: {folner_ok}")
results["folner_ok"] = float(folner_ok)

omega = np.exp(2j * np.pi * theta)
labels, A = setup(omega)
R3a = np.kron(np.roll(np.eye(3), 1, axis=0), np.eye(3))
R3b = np.kron(np.eye(3), np.roll(np.eye(3), 1, axis=0))
bad_counts, conc_ok = [], True
for N in [4, 5, 8, 13, 21, 34]:
    Aop = np.roll(np.eye(N), 1, axis=0).astype(complex)
    Bop = np.diag(omega ** np.arange(N))
    ops = {'a': Aop, 'A': Aop.conj().T, 'b': Bop, 'B': Bop.conj().T}
    good = []
    for i in range(N):
        e = np.zeros(N, complex); e[i] = 1
        ok = True
        for (row, col, c, w) in Gd['slots']:
            u_, v_ = z2_label(w)
            lhs = word_op(w, ops, N) @ e
            sec = np.linalg.matrix_power(Aop, u_ % N) @ (np.linalg.matrix_power(Bop, v_) if v_ >= 0 else
                                                        np.linalg.matrix_power(Bop.conj().T, -v_))
            if np.abs(lhs - twisted_phase(w, omega) * (sec @ e)).max() > 1e-10:
                ok = False; break
        if ok:
            good.append(i)
    k, kp = 9 * len(good), 9 * N
    Bs = {}
    for g in labels:
        u_, v_ = g
        sec = np.linalg.matrix_power(Aop, u_ % N) @ (np.linalg.matrix_power(Bop, v_) if v_ >= 0 else
                                                    np.linalg.matrix_power(Bop.conj().T, -v_))
        full = np.kron(sec, np.linalg.matrix_power(R3a, u_ % 3) @ np.linalg.matrix_power(R3b, v_ % 3))
        Bs[g] = sps.csr_matrix(full[:, [9 * i + t for i in good for t in range(9)]])
    iso, gram, tri = check_dilation(Bs, labels, A, omega, k)
    conc_ok = conc_ok and iso < 1e-10 and gram < 1e-9 and tri < 1e-10 and k == 9 * (N - 1)
    bad_counts.append(N - len(good))
print(f"concentrated model (x) Z_3^2, N = 4..34: bad points {bad_counts}; exact with k = 9(N-1), k' = 9N: {conc_ok}")
results["conc_bad_points"] = bad_counts
results["conc_ok"] = float(conc_ok)

Jphi = choi_from_kraus([A[g] for g in labels], d)
w_, V_ = np.linalg.eigh(Jphi)
Pi = V_[:, w_ > 1e-10] @ V_[:, w_ > 1e-10].conj().T
zr = twisted_phase('abAB', omega)
ratios, room_e2, ctd = [], {}, {}
for N in [5, 8, 12, 13, 20, 21, 34, 55, 89]:
    zeta = np.exp(2j * np.pi / N)
    pp = int(np.round(theta * N))
    S = np.roll(np.eye(N), 1, axis=0).astype(complex)
    Om = np.diag(zeta ** (pp * np.arange(N)))
    ops = {'a': S, 'A': S.conj().T, 'b': Om, 'B': Om.conj().T}
    Ms = [word_op(w, ops, N) for (_, _, _, w) in Gd['slots']]
    ns = len(Ms)
    Gram = np.array([[np.trace(Ms[b].conj().T @ Ms[a]) / N for b in range(ns)] for a in range(ns)])
    J = choi_from_slot_gram(Gd, Gram)
    leak = float(np.real(np.trace((np.eye(d * d) - Pi) @ J)))
    R = word_op('abAB', ops, N)
    e2 = float(np.real(np.trace((R - zr * np.eye(N)).conj().T @ (R - zr * np.eye(N))) / N))
    ratios.append(leak / e2)
    room_e2[N] = np.log2(N / (N - 1)) / e2
    ctd[N] = 0.5 * np.abs(np.linalg.eigvalsh(J - Jphi)).sum()
    print(f"thin round N={N:3d}: e^2={e2:.3e} leakage={leak:.3e} leakage/e^2={leak/e2:.4f} Choi-TD={ctd[N]:.3e} "
          f"room of any rectangular dilation at dim <= N >= {np.log2(N/(N-1)):.3e}, /e^2 = {room_e2[N]:.1f}")
results["thin_leak_over_e2_min"] = min(ratios)
results["thin_leak_over_e2_max"] = max(ratios)
results["thin_room_over_e2"] = [room_e2[13], room_e2[34], room_e2[89]]
results["thin_choi_td_89"] = ctd[89]
print(f"(total {time.time()-t00:.0f}s)")
print("RESULTS_JSON " + json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
