"""The spread-angle round and global scrambling (Theorem I.1). A: the family of Theorem I.1(a) is an exact rectangular dilation
with one bit of room and entropy production h2(0.6 sin^2 theta); forced growth 0.9997 bit at theta = 0.1. B: the device of
Theorem I.1(c) fills its active space. C: Lemma G.5 (mass outside the top-k eigenspace <= a ln 2). E: global Haar scrambling of
a biased memory multiplies its entropy production. Parts D and F print supplementary quantities the paper does not use.
Run: python -u check_frames.py
"""
import os
import sys
import json
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadget_lib import build_gadget, z2_label, kraus_from_gadget

results = {}


def h2(p):
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def Hev(ev):
    ev = ev[ev > 1e-15]
    return float(-(ev * np.log2(ev)).sum())


# ---------------------------------------------------------------- Part A
print("=== Part A: the spread-angle round ===")
k = 7
U = np.roll(np.eye(k), 1, axis=0)


def fam(t, k=7):
    U = np.roll(np.eye(k), 1, axis=0)
    c, s = np.cos(t), np.sin(t)
    return [np.vstack([np.eye(k), np.zeros((k, k))]), np.vstack([c * U, s * U]), np.vstack([c * U.T, -s * U.T])]


a_err, struct_err, rooms, kyfan, costs = 0.0, 0.0, [], {}, []
for t in [0.3, 0.1, 0.03, 0.01, 0.003]:
    B1, Bx, Bxi = fam(t)
    Bs = (B1, Bx, Bxi)
    struct_err = max(struct_err, max(np.abs(B.T @ B - np.eye(k)).max() for B in Bs),
                     np.abs(B1.T @ Bxi - Bx.T @ B1).max(),
                     np.abs(np.array([[np.trace(P.T @ Q) / k for Q in Bs] for P in Bs]) - np.eye(3)).max())
    T = (0.4 * B1 @ B1.T + 0.3 * Bx @ Bx.T + 0.3 * Bxi @ Bxi.T) / k
    a = Hev(np.linalg.eigvalsh(T)) - np.log2(k)
    a_err = max(a_err, abs(a - h2(0.6 * np.sin(t) ** 2)))
    rooms.append(np.log2(np.linalg.matrix_rank(np.hstack(Bs), tol=1e-9) / k))
    kyfan[t] = np.log2(max(1.0, 2 - 2 * 1e-6 / (1 - np.cos(t))))
    s2 = np.sin(t) ** 2
    costs.append((h2(s2) + np.log2(1 + s2)) / h2(0.6 * s2))
    print(f"theta={t:5.3f}: a={a:.6e} (h2(0.6 sin^2) = {h2(0.6*np.sin(t)**2):.6e}), rank room {rooms[-1]:.3f} bit, "
          f"Ky Fan forced growth at eta=1e-6: {kyfan[t]:.4f} bit, path cost / a = {costs[-1]:.3f}")
results["sa_struct_err"] = struct_err
results["sa_a_err"] = a_err
results["sa_rank_room"] = rooms
results["sa_kyfan_theta0.1"] = kyfan[0.1]
results["sa_path_cost_over_a"] = costs

# ---------------------------------------------------------------- Part B
print("\n=== Part B: the global-Haar device of Theorem I.1(c) on the spread-angle family (k = 4) ===")


def frame_run(rng, thetas, k=4, m0=2, T=4):
    mu = {0: 0.4, 1: 0.3, 2: 0.3}

    def haar(n):
        z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
        q, r = np.linalg.qr(z)
        return q * (np.diag(r) / abs(np.diag(r)))

    def smooth_rank(sig, eta):
        ev = np.sort(np.linalg.eigvalsh(sig))[::-1]
        return int(np.searchsorted(np.cumsum(ev), 1 - eta) + 1)

    out = {}
    for theta in thetas:
        Bs = fam(theta, k)
        D0 = k * m0
        imgs = {(): np.eye(D0, dtype=complex)}
        m = m0
        rows = []
        for t in range(1, T + 1):
            V = haar(k * m)
            new = {}
            for y, Y in imgs.items():
                Z = V @ Y
                for g, B in enumerate(Bs):
                    new[y + (g,)] = np.kron(B, np.eye(m)) @ Z
            imgs, m = new, 2 * m
            Dt = k * m
            su = sum(Y @ Y.conj().T for Y in imgs.values()) / (len(imgs) * D0)
            smu = sum(np.prod([mu[g] for g in y]) * (Y @ Y.conj().T) for y, Y in imgs.items()) / D0
            rows.append((Dt, smooth_rank(su, 1e-6) / Dt, Hev(np.linalg.eigvalsh(smu)) - np.log2(D0)))
        out[theta] = rows
    return out


res = frame_run(np.random.default_rng(7), [0.3, 0.1])
for theta in [0.3, 0.1]:
    p = 0.6 * np.sin(theta) ** 2
    print(f"theta={theta}: a = {h2(p):.4f} bit/round; per t = 1..4: "
          + "; ".join(f"D_t={Dt} rank_1e-6/D_t={rf:.3f} S(sigma_mu)-log D0={ds:.3f}" for (Dt, rf, ds) in res[theta]))
results["growth_entropy_theta0.1"] = [r_[2] for r_ in res[0.1]]
results["growth_rank_fraction_theta0.1"] = [r_[1] for r_ in res[0.1]]
growth = []
for m0 in [1, 2, 4, 8]:
    rr = frame_run(np.random.default_rng(7), [0.1], m0=m0)[0.1]
    growth.append(rr[-1][2])
    print(f"D0 = {4*m0:2d}: S(sigma_mu) - log D0 at t = 4: {rr[-1][2]:.3f} (t*a = {4*h2(0.6*np.sin(0.1)**2):.3f})")
results["growth_entropy_t4_D0sweep"] = growth

# ---------------------------------------------------------------- Part C
print("\n=== Part C: top-eigenspace lemma ===")
rng = np.random.default_rng(3)
worst = 0.0
for trial in range(3000):
    kk = int(rng.integers(2, 7)); N = kk * int(rng.integers(2, 4)); rr_ = int(rng.integers(2, 5))
    mu_ = rng.dirichlet(np.ones(rr_))
    eps = 10 ** rng.uniform(-4, 0)
    base = np.linalg.qr(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))[0][:, :kk]
    T = np.zeros((N, N), dtype=complex)
    for g in range(rr_):
        Z = base + eps * (rng.normal(size=(N, kk)) + 1j * rng.normal(size=(N, kk)))
        Q = np.linalg.qr(Z)[0][:, :kk]
        T += mu_[g] * (Q @ Q.conj().T) / kk
    ev = np.sort(np.linalg.eigvalsh(T))[::-1]
    a = Hev(ev) - np.log2(kk); w = ev[kk:].sum()
    if a > 1e-12:
        worst = max(worst, w / (a * np.log(2)))
print(f"max over 3000 random families of w/(a ln 2) = {worst:.4f} (the lemma says <= 1)")
results["top_eigenspace_max_ratio"] = worst

# ---------------------------------------------------------------- Part D
print("\n=== Part D: factorization of the spread-angle family ===")
fac_err, gar = 0.0, []
for t in [0.05, 0.2, 0.7]:
    c, s = np.cos(t), np.sin(t)
    B = dict(zip([1, 'x', 'xi'], fam(t)))
    W = {1: np.eye(k), 'x': U, 'xi': U.T}
    gam = {1: np.array([1.0, 0]), 'x': np.array([c, s]), 'xi': np.array([c, -s])}
    fac_err = max(fac_err, max(np.abs(B[g] - np.kron(gam[g].reshape(2, 1), W[g])).max() for g in B),
                  np.abs(W[1].T @ W['xi'] - W['x'].T @ W[1]).max())
    om = sum(mu * np.outer(gam[g], gam[g]) for g, mu in [(1, 0.4), ('x', 0.3), ('xi', 0.3)])
    gar.append(abs(Hev(np.linalg.eigvalsh(om)) - h2(0.6 * s * s)))
    print(f"t={t}: factorization error {fac_err:.1e}; garbage entropy {Hev(np.linalg.eigvalsh(om)):.6f} "
          f"= h2(0.6 sin^2 t) = {h2(0.6*s*s):.6f}")
results["factor_err"] = fac_err
results["garbage_entropy_err"] = max(gar)

# ---------------------------------------------------------------- Part E
print("\n=== Part E: global Haar scrambling of a biased memory ===")
rng = np.random.default_rng(3)


def haar_e(n):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    Q, R_ = np.linalg.qr(Z)
    return Q * (np.diag(R_) / np.abs(np.diag(R_)))


def S_(rho):
    w = np.linalg.eigvalsh((rho + rho.conj().T) / 2); w = w[w > 1e-14]
    return float(-(w * np.log2(w)).sum())


mu = {0: 0.4, 1: 0.3, -1: 0.3}
scr = {}
for (kk, sig, m) in [(16, 3.0, 4), (16, 3.0, 8), (24, 4.0, 4), (24, 4.0, 6), (32, 6.0, 4)]:
    kp = kk + 2
    Sh = {}
    for j in mu:
        Mm = np.zeros((kp, kk))
        for x in range(kk):
            Mm[x + 1 + j, x] = 1
        Sh[j] = Mm
    x = np.arange(kk); p = np.exp(-(x - (kk - 1) / 2) ** 2 / (2 * sig ** 2)); p /= p.sum()
    rho = np.kron(np.diag(p), np.eye(m) / m)

    def call(s):
        return sum(mu[j] * np.kron(Sh[j], np.eye(m)) @ s @ np.kron(Sh[j], np.eye(m)).T for j in mu)

    a_un = S_(call(rho)) - S_(rho)
    incs = []
    for _ in range(12):
        V = haar_e(kk * m)
        sV = V @ rho @ V.conj().T
        incs.append(S_(call(sV)) - S_(sV))
    a_flat = S_(sum(mu[j] * Sh[j] @ (np.eye(kk) / kk) @ Sh[j].T for j in mu)) - np.log2(kk)
    scr[(kk, m)] = (a_un, float(np.mean(incs)), a_flat)
    print(f"k={kk} sigma={sig} m={m}: a(rho) = {a_un:.4e}; after global Haar {np.mean(incs):.4e} +- {np.std(incs):.1e} "
          f"(ratio {np.mean(incs)/a_un:.1f}); a_flat = {a_flat:.4e}; log2(k'/k) = {np.log2(kp/kk):.4e}")
results["scramble_ratios"] = [scr[(16, 4)][1] / scr[(16, 4)][0], scr[(24, 4)][1] / scr[(24, 4)][0],
                              scr[(32, 4)][1] / scr[(32, 4)][0]]
results["scramble_over_flat_min"] = min(v[1] / v[2] for v in scr.values())

# ---------------------------------------------------------------- Part F
print("\n=== Part F: history collisions ===")
# (i) exact: the Z gadget with a cyclic representation reused on one memory, anchor superposition inputs
Gd = build_gadget(['a'], [])
d = Gd['d']
labs, A = kraus_from_gadget(Gd, z2_label, lambda w: 1.0)
N = 7
Ssh = np.roll(np.eye(N), 1, axis=0)
lam = {g: np.linalg.matrix_power(Ssh, g[0] % N) for g in labs}
W = sum(np.kron(A[g], lam[g]) for g in labs)
assert np.allclose(W.conj().T @ W, np.eye(d * N))
z1, za, zA = 0, 1, 2                      # anchor positions of the labels 1, a, A
psi1 = np.zeros(d); psi1[[za, z1]] = 1 / np.sqrt(2)
psi2 = np.zeros(d); psi2[[zA, z1]] = 1 / np.sqrt(2)
sigma = np.eye(N) / N
st = W @ np.kron(np.outer(psi1, psi1), sigma) @ W.conj().T                      # out1 (x) mem
st = np.kron(np.outer(psi2, psi2), st)                                           # in2 (x) out1 (x) mem
st = st.reshape(d, d, N, d, d, N).transpose(1, 0, 2, 4, 3, 5).reshape(d * d * N, d * d * N)  # out1, in2, mem
W2 = np.kron(np.eye(d), W)
st = W2 @ st @ W2.conj().T
obs = st.reshape(d * d, N, d * d, N).trace(axis1=1, axis2=3)
Phi = lambda X: sum(A[g] @ X @ A[g].conj().T for g in labs)
ideal = np.kron(Phi(np.outer(psi1, psi1)), Phi(np.outer(psi2, psi2)))
err_exact = 0.5 * np.abs(np.linalg.eigvalsh(obs - ideal)).sum()
print(f"Z gadget, cyclic representation on C^{N} reused twice, anchor-superposition observer: error = {err_exact:.6f}")
results["collision_exact_error"] = err_exact

# (ii) twisted-Z^2 round on a 16x16 box (x) Z_3^2, non-diagonal bath, no re-randomization, flat-probe observer
theta = (np.sqrt(5) - 1) / 2
omega = np.exp(2j * np.pi * theta)


def letter_maps(Lb):
    maps = {}
    for c in 'aAbB':
        src, dst, ph = [], [], []
        for i in range(Lb):
            for j in range(Lb):
                if c == 'a': ni, nj, p = i + 1, j, 1.0
                elif c == 'A': ni, nj, p = i - 1, j, 1.0
                elif c == 'b': ni, nj, p = i, j + 1, omega ** i
                else: ni, nj, p = i, j - 1, np.conj(omega ** i)
                if 0 <= ni < Lb and 0 <= nj < Lb:
                    src.append(i * Lb + j); dst.append(ni * Lb + nj); ph.append(p)
        maps[c] = (np.array(src), np.array(dst), np.array(ph))
    return maps


def word_matrix(word, maps, N):
    Mw = np.eye(N, dtype=complex)
    for c in word:
        src, dst, ph = maps[c]
        L = np.zeros((N, N), complex); L[dst, src] = ph
        Mw = Mw @ L
    return Mw


def history_collision_error(Lb=16, sigma=2.0, kappa=0.6, n=2):
    import itertools
    N = Lb * Lb
    maps = letter_maps(Lb)
    ii, jj = np.divmod(np.arange(N), Lb)
    cpos = (Lb - 1) / 2
    Vp = ((ii - cpos) ** 2 + (jj - cpos) ** 2) / (2 * sigma ** 2)
    hop = sum(word_matrix(c, maps, N) for c in 'aAbB')
    Hm = np.diag(Vp) - kappa * hop; Hm = (Hm + Hm.conj().T) / 2
    w, Uh = np.linalg.eigh(Hm); p = np.exp(-(w - w.min())); p /= p.sum()
    rho = (Uh * p) @ Uh.conj().T
    labs2 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1)]
    sec = {}
    for (u, v) in labs2:
        word = ('a' * u if u >= 0 else 'A' * (-u)) + ('b' * v if v >= 0 else 'B' * (-v))
        sec[(u, v)] = word_matrix(word, maps, N)
    hist = list(itertools.product(labs2, repeat=n))
    ops, zs = [], []
    for y in hist:
        Mh = np.eye(N, dtype=complex)
        for g in y:
            Mh = sec[g] @ Mh
        ops.append(Mh)
        zs.append((sum(g[0] for g in y) % 3, sum(g[1] for g in y) % 3))
    R_ = len(hist)
    Gm = np.zeros((R_, R_), complex)
    for x in range(R_):
        for y in range(R_):
            if zs[x] == zs[y]:
                Gm[x, y] = np.trace(rho @ ops[x].conj().T @ ops[y])
    ev = np.linalg.eigvalsh((Gm + Gm.conj().T) / 2 - np.eye(R_))
    return 0.5 * np.abs(ev).sum() / R_, np.abs(ev).max(), R_


coll = []
for n in [1, 2, 3]:
    err, opn, R_ = history_collision_error(n=n)
    coll.append(err)
    print(f"twisted Z^2, bath reused without scrambling, n={n}: histories {R_}, flat-probe error {err:.4f}, "
          f"||Gram - I|| = {opn:.3f}")
results["collision_twisted_errors"] = coll
print("RESULTS_JSON " + json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
