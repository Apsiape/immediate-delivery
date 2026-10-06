"""Finite controls for Remark 2.5 (proof in Appendix F.8): for one anchored triangle, the gadget unitary built from near-
representations has zero production, leakage at most the squared slot defect and Choi distance at most the defect; padding has
real nonnegative moments; t = ceil(ln(32 d (r-1))/ln(4/3)) and c = 1/(32 d t) meet u <= 3/64 and leakage <= 2/(15 n). Not a proof.
"""
from pathlib import Path
import math
import re
import numpy as np


def main():
    count = 0

    def check(ok, label):
        nonlocal count
        count += 1
        if not ok:
            raise AssertionError(label)

    # One anchored triangle a*b=c, with four distinct labels.
    d, r = 6, 4
    slots = [(0, 0, 0, 1), (1, 1, 1, 1), (2, 2, 2, 1),
             (3, 3, 3, 1), (4, 4, 0, 1/math.sqrt(2)),
             (4, 5, 2, 1/math.sqrt(2)), (5, 4, 1, 1/math.sqrt(2)),
             (5, 5, 3, -1/math.sqrt(2))]
    a = [np.zeros((d, d)) for _ in range(r)]
    for i, j, g, coeff in slots:
        a[g][i, j] = coeff
    vectors = np.stack([v.reshape(-1) for v in a])
    masses = (vectors * vectors).sum(axis=1)
    projector = sum(np.outer(v, v)/m for v, m in zip(vectors, masses))
    target = sum(np.outer(v, v) for v in vectors)/d
    rng = np.random.default_rng(169)
    for k in (2, 3):
        eye = np.eye(k)
        for angle in (0, .001, .01, .1):
            va = np.linalg.qr(rng.normal(size=(k, k)) +
                              1j*rng.normal(size=(k, k)))[0]
            vb = np.linalg.qr(rng.normal(size=(k, k)) +
                              1j*rng.normal(size=(k, k)))[0]
            perturb = np.diag(np.exp(1j*angle*np.linspace(-1, 1, k)))
            vc = va @ vb @ perturb
            vs = [eye, va, vb, vc]
            blocks = [[np.zeros((k, k), complex) for _ in range(d)]
                      for _ in range(d)]
            for i, j, g, coeff in slots:
                blocks[i][j] = coeff * (va @ vb if (i, j) == (5, 5) else vs[g])
            u = np.block(blocks)
            check(np.linalg.norm(u.conj().T @ u-np.eye(d*k)) < 1e-12,
                  "physical block unitary")
            bath = sum(b @ b.conj().T for row in blocks for b in row)/(d*k)
            check(np.linalg.norm(bath-eye/k) < 1e-12, "actual full-flat gain zero")
            actual = [blocks[i][j] for i in range(d) for j in range(d)]
            ideal = [sum(v[i, j]*b for v, b in zip(a, vs))
                     for i in range(d) for j in range(d)]
            choi = np.array([[np.trace(x @ y.conj().T)/(d*k) for y in actual]
                             for x in actual])
            gram = np.array([[np.trace(x @ y.conj().T)/(d*k) for y in ideal]
                             for x in ideal])
            residual = sum(np.linalg.norm(x-y)**2
                           for x, y in zip(actual, ideal))/(d*k)
            defect2 = np.linalg.norm(va @ vb-vc)**2/k
            leakage = float(np.trace((np.eye(d*d)-projector) @ choi).real)
            check(abs(residual-defect2/(2*d)) < 1e-12, "slot coefficient and norm")
            check(leakage <= residual+1e-12, "quadratic support leakage")
            check(np.linalg.svd(choi-gram, compute_uv=False).sum()/2 <=
                  math.sqrt(residual)+1e-12, "Choi vector comparison")
            z = max(abs(np.trace(v.conj().T @ w)/k)
                    for i, v in enumerate(vs) for j, w in enumerate(vs) if i != j)
            check(np.linalg.svd(gram-target, compute_uv=False).sum()/2 <=
                  (r-1)*z/2+1e-12, "weighted moment trace bound")
            for v, w in zip(vs[:-1], vs[1:]):
                padding_v = np.block([[v, np.zeros((k, 3*k))],
                                      [np.zeros((k, k)), v.conj(), np.zeros((k, 2*k))],
                                      [np.zeros((2*k, 2*k)), np.eye(2*k)]])
                padding_w = np.block([[w, np.zeros((k, 3*k))],
                                      [np.zeros((k, k)), w.conj(), np.zeros((k, 2*k))],
                                      [np.zeros((2*k, 2*k)), np.eye(2*k)]])
                expected = (1+np.trace(v.conj().T @ w).real/k)/2
                moment = np.trace(padding_v.conj().T @ padding_w)/(4*k)
                check(abs(moment-expected) < 1e-12, "real nonnegative padded moment")
                check(abs(np.trace(np.kron(padding_v, padding_v).conj().T @
                                   np.kron(padding_w, padding_w))/(4*k)**2 -
                          expected**2) < 1e-12, "tensor moment powers")

    for d, r in ((6, 4), (104, 26), (200, 40)):
        t = math.ceil(math.log(32*d*(r-1))/math.log(4/3))
        c = 1/(32*d*t)
        for n in (1, 2, 10, 10000):
            e = c/math.sqrt(n)
            z = (3/4)**t
            check(d*(t*e+(r-1)*z/2) <= 3/64+1e-15, "coarse-error budget")
            check((t*e)**2 <= 2/(15*n), "profile leakage budget")
    print(f"{count} checks for Remark 2.5 passed (finite controls, not a proof).")


if __name__ == "__main__":
    main()
