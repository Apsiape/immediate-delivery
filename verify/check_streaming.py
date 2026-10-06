"""Robust streaming, Theorem 7.6(a), on the Z gadget (d = 5) with families that have exact triangle identities and only a
flat-Gram defect: the per-round comparison d(Phi_W, Phi) <= ||G - I||/2 from its proof (attained by family B), product
observers over n rounds below (n/2)||G - I||, and the Haar-mean step E[V*(B_g* B_h (x) I)V] = tau(B_g* B_h) I.
Run: python -u check_streaming.py
"""
import os
import sys
import json
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gadget_lib import build_gadget, z2_label, kraus_from_gadget

results = {}
Gd = build_gadget(['a'], [])
d = Gd['d']
labels, A = kraus_from_gadget(Gd, z2_label, lambda w: 1.0)
labels = sorted(labels)            # (-1,0) = A, (0,0) = 1, (1,0) = a
K = [A[g] for g in labels]
r = len(K)


def shift_model(blocks):
    k = sum(blocks)
    S = np.zeros((k, k))
    o = 0
    for m in blocks:
        for i in range(m):
            S[o + (i + 1) % m, o + i] = 1
        o += m
    return {g: np.linalg.matrix_power(S, g[0] % (np.lcm.reduce(blocks))) for g in labels}, k


def gram(B, k):
    return np.array([[np.trace(B[g].T @ B[h]) / k for h in labels] for g in labels])


def comp_output(rho_in):
    return np.array([[np.trace(rho_in @ K[h].conj().T @ K[g]) for h in range(r)] for g in range(r)])


def nfold_product_error(P, Gm, n):
    """half trace distance of the observer states after n rounds of a product strategy whose per-round
    complementary Gram is P: (P^1/2 G^T P^1/2)^(x)n against P^(x)n."""
    w, V = np.linalg.eigh((P + P.conj().T) / 2)
    Ph = (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T
    X = Ph @ Gm.T @ Ph
    Y = (P + P.conj().T) / 2
    Xn, Yn = np.array([[1.0 + 0j]]), np.array([[1.0 + 0j]])
    for _ in range(n):
        Xn = np.kron(Xn, X); Yn = np.kron(Yn, Y)
    Dm = Xn - Yn
    return 0.5 * np.abs(np.linalg.eigvalsh((Dm + Dm.conj().T) / 2)).sum()


rng = np.random.default_rng(7)
rho_flat = np.zeros((d, d)); rho_flat[0, 0] = rho_flat[1, 1] = rho_flat[2, 2] = 1 / 3
psi_pair = np.zeros(d); psi_pair[[1, 2]] = 1 / np.sqrt(2)
ratios, worst_slack = [], np.inf
for fam, base in [('R', 1), ('N', 2)]:
    for q in [1, 3, 9]:
        B, k = shift_model([base] + [3] * q)
        W = sum(np.kron(K[i], B[labels[i]]) for i in range(r))
        iso = np.abs(W.conj().T @ W - np.eye(d * k)).max()
        Gm = gram(B, k)
        gnorm = np.abs(np.linalg.eigvalsh(Gm - np.eye(r))).max()
        best_val, best_P = -1, None
        for trial in range(4000):
            v = rng.normal(size=d) + 1j * rng.normal(size=d); v /= np.linalg.norm(v)
            P = comp_output(np.outer(v, v.conj()))
            val = nfold_product_error(P, Gm, 1)
            if val > best_val:
                best_val, best_P = val, P
        Pflat = comp_output(rho_flat)
        one_flat = nfold_product_error(Pflat, Gm, 1)
        one_pair = nfold_product_error(comp_output(np.outer(psi_pair, psi_pair)), Gm, 1)
        print(f"[{fam} q={q}] |W*W-I|={iso:.1e} ||G-I||={gnorm:.4f}: one round, flat probe {one_flat:.5f}, "
              f"pair input {one_pair:.5f}, bound ||G-I||/2 = {gnorm/2:.5f}")
        results[f"{fam}{q}_gnorm"] = gnorm
        results[f"{fam}{q}_one_flat"] = one_flat
        results[f"{fam}{q}_one_pair"] = one_pair
        flat_vals, best_vals = [], []
        for n in [1, 2, 4, 6]:
            ef = nfold_product_error(Pflat, Gm, n)
            eb = nfold_product_error(best_P, Gm, n)
            flat_vals.append(ef); best_vals.append(eb)
            worst_slack = min(worst_slack, n * gnorm / 2 - max(ef, eb))
            if n > 1:
                ratios.append(eb / (np.sqrt(n) * best_val))
        print(f"      n = 1, 2, 4, 6: flat probe {[round(float(x), 5) for x in flat_vals]}, best random input "
              f"{[round(float(x), 5) for x in best_vals]}, bound {[round(float(n * gnorm / 2), 5) for n in [1, 2, 4, 6]]}")
        results[f"{fam}{q}_flat_n"] = flat_vals
results["product_ratio_min"] = min(ratios)
results["product_ratio_max"] = max(ratios)
results["bound_slack_min"] = worst_slack
print(f"best product observer / (sqrt(n) * one round), n = 2..6: {min(ratios):.3f} .. {max(ratios):.3f}; "
      f"smallest slack (n/2)||G-I|| - value = {worst_slack:.4f}")

# ---------------------------------------------------------------- Part C: Haar mean
rng = np.random.default_rng(7)


def haar(n):
    Z = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.sqrt(2)
    Q, R_ = np.linalg.qr(Z)
    return Q * (np.diag(R_) / np.abs(np.diag(R_)))


B, k = shift_model([1, 3])
Gm = gram(B, k)
mfac, Sdim, ncol = 3, 4, 3
D = k * mfac
F = (rng.normal(size=(D * Sdim, ncol)) + 1j * rng.normal(size=(D * Sdim, ncol))) / np.sqrt(D * Sdim)
FF = F.conj().T @ F
acc = np.zeros((r * ncol, r * ncol), complex)
ns = 4000
for _ in range(ns):
    V = haar(D)
    VF = np.kron(V, np.eye(Sdim)) @ F
    Fn = np.hstack([np.kron(np.kron(B[g], np.eye(mfac)), np.eye(Sdim)) @ VF for g in labels])
    acc += Fn.conj().T @ Fn
acc /= ns
pred = np.block([[Gm[i, j] * FF for j in range(r)] for i in range(r)])
dev = np.abs(acc - pred).max()
print(f"Haar mean of the Gram blocks after one call ({ns} samples, D = {D}): max deviation {dev:.4f}, "
      f"largest entry {np.abs(pred).max():.4f}")
results["haar_mean_dev"] = dev
results["haar_mean_scale"] = np.abs(pred).max()
print("RESULTS_JSON " + json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
