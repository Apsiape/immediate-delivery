"""Finite controls for Theorem Q.3 (entropy without source symmetry): the constant c = 1/(1280 * 2 sqrt(104 ln 2)), the bracket
7/12 - 7/128 - f6(9/16384) > 7/16, exact S_{R+1} reflection normals (A_R Gram), the clock-mass and Araki-Lieb steps and the
rank-six Holevo bound. Consistency checks, not a proof.
"""
from pathlib import Path
import math
import numpy as np


COUNT = 0


def check(condition, label):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(label)


def entropy(a):
    p = np.linalg.eigvalsh((a + a.conj().T) / 2)
    p = p[p > 1e-13]
    return float(-np.dot(p, np.log2(p)))


def root(a):
    p, v = np.linalg.eigh((a + a.conj().T) / 2)
    return (v * np.sqrt(np.maximum(p, 0))) @ v.conj().T


def state(rng, dim):
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    z = z @ z.conj().T
    return z / np.trace(z)


def f6(x):
    if x == 0:
        return 0.0
    return -x * math.log2(x) - (1-x) * math.log2(1-x) + x * math.log2(5)


def normals(rng, m, r, i0, j0):
    """Direct inverse-prefix columns, with sparse eligible-edge phases.

    These samples use the actual region supports but need not realize
    constant curl. The Gram/support claim is independent of their angles.
    No whole orbital twirl or enormous exterior algebra is constructed.
    """
    steps, pairs = [], {}
    for q in range(r-1):
        i = i0 + q
        rotations = []
        if j0:
            for t in range(4, m+4):
                eligible = i+j0 <= (m-3 if t == 4 else t+m-8)
                if not eligible or rng.random() < .5:
                    continue
                x, y = t+m-1-i-j0, t+2*m-1-j0
                check(x >= 5 and y > 2*m+2-i-j0, "interior mode bounds")
                check(x+q not in pairs or pairs[x+q] == y, "fixed start pair")
                pairs[x+q] = y
                rotations.append((x-1, y-1, float(rng.uniform(-math.pi, math.pi))))
        steps.append((2*m+2-i-j0, rotations))
    answer = []
    for j in range(r):
        v = np.zeros(4*m+4, complex)
        v[0], v[1] = 1/math.sqrt(2), -1/math.sqrt(2)
        # P_j* acts through the last edge first: inverse A, then inverse U.
        for n, rotations in reversed(steps[:j]):
            v[:n] = np.roll(v[:n], 1)
            for x, y, theta in rotations:
                z = (np.exp(-1j*theta)-1) * (v[x]-v[y]) / 2
                v[x] += z
                v[y] -= z
        allowed = {j, j+1}
        for x in (j+1, j+2):
            if x in pairs:
                allowed.add(pairs[x]-1)
        outside = v.copy()
        outside[list(allowed)] = 0
        check(np.linalg.norm(outside) < 1e-12, "normal has only unique far partners")
        answer.append(v)
    return np.array(answer)


def main():
    rng = np.random.default_rng(165)
    cg = 2*math.sqrt(104*math.log(2))
    c = 1/(1280*cg)
    check(4.60e-5 < c < 4.61e-5, "safe entropy coefficient")
    check(7/12-7/128-f6(9/16384) > 7/16, "entropy bracket")
    for r in range(5, 129):
        check((r+1)//3 >= r/4, "S3 block count")
        check(2*r+5 <= 3*r, "six-word lengths")
    for k in (0, 127.9, 128, 639.9, 640, 1024, 1e8):
        r = math.floor(k/128)
        lower = 1 if r < 5 else 1+r/10
        check(lower >= k/1280, "small-R and rounding ledger")

    for m, r in ((48, 8), (640, 5), (1024, 8)):
        for i0, j0 in ((0, 0), (0, 1), (m//2, m//2),
                       (m-r-1, 1), (m-r-1, m-r-1)):
            v = normals(rng, m, r, i0, j0)
            gram = v.conj() @ v.T
            target = np.eye(r)-.5*np.eye(r, k=1)-.5*np.eye(r, k=-1)
            check(np.linalg.norm(gram-target) < 1e-12, "direct prefix A_R Gram")

    for m in (8, 48):
        for _ in range(8):
            p = rng.random(m)
            p /= p.sum()
            e = np.square(np.roll(np.sqrt(p), 1)-np.sqrt(p)).sum()
            check(p.max() <= 1/m+2*math.sqrt(e), "coordinate mass bound")

    for _ in range(12):
        rho = state(rng, 9)
        blocks = [rho[3*i:3*i+3, 3*i:3*i+3] for i in range(3)]
        conditional = sum(float(np.trace(b).real)*entropy(b/np.trace(b)) for b in blocks)
        check(entropy(rho)+1e-12 >= conditional, "coherent-clock conditional entropy")

    for dim in (3, 5):
        for delta in (.01, .05, .1):
            rho = state(rng, dim)
            xi = root(rho)
            roots, outputs = [], []
            for _ in range(6):
                h = state(rng, dim)
                p, v = np.linalg.eigh(h)
                u = (v*np.exp(1j*delta*p)) @ v.conj().T
                roots.append((u @ xi @ u.conj().T).reshape(-1))
                outputs.append(u @ rho @ u.conj().T)
            vectors = np.array(roots)
            x = float(max(np.linalg.norm(z-xi.reshape(-1))**2 for z in vectors))
            omega_entropy = entropy(vectors.conj() @ vectors.T / 6)
            gain = entropy(sum(outputs)/6)-entropy(rho)
            check(x <= 5/6, "rank-six range")
            check(gain <= omega_entropy+1e-11, "root-ensemble data processing")
            check(omega_entropy <= f6(x)+1e-11, "rank-six entropy ceiling")

    print(f'{COUNT} checks for Theorem Q.3 passed (finite controls, not a proof).')


if __name__ == '__main__':
    main()
