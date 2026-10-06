"""Finite controls for Theorem R.1 (moderate flavour growth): the constants delta, c_f, b, kappa and c_0, the block geometry
(density, block count, sweep length, angle spread 54 pi/M^3), the S_4 frame and A_4 charge, the six-mode gap, the contraction
I - T*T >= t^2 (1 - t^2 h^2/3) A_2, and the cubic two-branch cost algebra. Consistency checks, not a proof.
"""
from pathlib import Path
import itertools
import math
import numpy as np


COUNT = 0


def check(condition, label):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(label)


def exp_h(h, t):
    values, vectors = np.linalg.eigh(h)
    return (vectors * np.exp(1j*t*values)) @ vectors.conj().T


def kron_power(a, k):
    out = np.ones((1, 1))
    for _ in range(k):
        out = np.kron(out, a)
    return out


def tensor_generator(h, k):
    size = h.shape[0]
    out = np.zeros((size**k, size**k), complex)
    for place in range(k):
        term = np.ones((1, 1))
        for j in range(k):
            term = np.kron(term, h if j == place else np.eye(size))
        out += term
    return out


def main():
    delta = 2/(12*(45*math.pi)**2)
    cf = delta*math.pi/5184
    b = delta*math.pi**2/24
    kappa = b/655360
    e0 = min(1/480, kappa/8)
    cg = 2*math.sqrt(104*math.log(2))
    c0 = min(e0**2/cg**2, 488*kappa/(225*104))
    check(c0 > 0 and 0 < cf < 1/64, "nonvacuous asymptotic constants")
    check(216*math.pi*cf <= delta*math.pi**2/24*(1+1e-14),
          "unequal-phase error at most half reference gap")
    check(4.5*math.pi*cf < 1, "generator bandwidth condition")

    for m in list(range(4096, 4160))+[8192, 16384, 100000, 1000000]:
        x, z = m//4, m//256
        j_count = (z-1)//8
        count = (m-x-11)*(m+x-48-4*z)//2
        check(count/m**2 >= 15/32-40/m-1/128 >= .45, "good wedge density")
        check(m/8192 <= j_count <= m/2048, "linear number of blocks")
        start = x-z
        start += start % 2
        last = start+8*(j_count-1)+6
        check(last+1 <= x, "blocks within exact logical chain")
        check(2*(x+8)+420*j_count <= m, "product-group sweep length")
        for i in [0, (m-x-12)//2, m-x-12]:
            lower = m-x+16+2*z-i
            for j in [lower, m-3]:
                angles = []
                for s in range(start, start+8, 2):
                    t = s-m+1+i+j
                    h = m+3-t
                    area = (m-1)**2-h*(h+1)//2
                    check(z+17 <= t <= m-16, "active untruncated bulk region")
                    check(m*m/3 <= area <= m*m, "area bounds")
                    check(i+j <= t+m-8, "selected field active")
                    angles.append(-(-1)**(t-5)*math.pi/area)
                check(all(a*angles[0] > 0 for a in angles), "same signs")
                check(max(abs(a-angles[0]) for a in angles) <= 54*math.pi/m**3,
                      "within-block angle error")
        power = math.isqrt(m**3)
        check(m**1.5/2 <= power <= m**1.5, "integer power bounds")

    # Every selected-four permutation fixes the holes in eight points.
    for perm in itertools.permutations(range(4)):
        target = list(range(8))
        for j, image in enumerate(perm):
            target[2*j] = 2*image
        work, swaps = list(range(8)), []
        for j in range(8):
            index = work.index(target[j])
            while index > j:
                work[index-1], work[index] = work[index], work[index-1]
                swaps.append(index-1)
                index -= 1
        check(work == target and len(swaps) <= 28, "local S8 adjacent word")
        check(sum(2*q+1 for q in swaps) <= 420, "literal local word budget")

    # Exact S4 standard frame and its six-mode generators.
    q = np.array([[1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
                  [-1/math.sqrt(2), 1/math.sqrt(6), 1/math.sqrt(12)],
                  [0, -2/math.sqrt(6), 1/math.sqrt(12)],
                  [0, 0, -3/math.sqrt(12)]])
    generators, actions, parity, reps = [], [], [], []
    for perm in itertools.permutations(range(4)):
        matrix = np.eye(4)[:, perm]
        r = q.T @ matrix @ q
        reps.append(r)
        generators.append(-.5*np.block([[np.zeros((3, 3)), r],
                                        [r.T, np.zeros((3, 3))]]))
        actions.append(np.block([[r, np.zeros((3, 3))],
                                 [np.zeros((3, 3)), np.eye(3)]]))
        parity.append(round(np.linalg.det(matrix)))
    frame = np.array(reps).reshape(24, 9)
    check(np.linalg.norm(frame.mean(axis=0)) < 1e-13, "zero mean standard frame")
    check(np.linalg.norm(frame.T @ frame/24-np.eye(9)/3) < 1e-13,
          "matrix coefficient orthogonality")
    for a in range(3):
        for bb in range(3):
            coordinate = -6*sum(r[a, bb]*h for r, h in zip(reps, generators))/24
            expected = np.zeros((6, 6))
            expected[a, 3+bb] = expected[3+bb, a] = 1
            check(np.linalg.norm(coordinate-expected) < 1e-13,
                  "coordinate generator normalization")

    # Degree-one tensor sectors, not exhaustive exterior representations.
    for k in [1, 2]:
        hs = [tensor_generator(h, k) for h in generators]
        charge = np.eye(6**k)-sum(kron_power(g, k) for g, sign in zip(actions, parity)
                                 if sign == 1)/12
        a2 = sum(h @ h for h in hs)/24
        check(np.linalg.norm(charge @ charge-charge) < 1e-12, "A4 charge projection")
        check(np.linalg.eigvalsh(a2-delta*charge).min() >= -1e-12,
              "infinitesimal gap finite control")
        for t in [.01, .03, .1]:
            average = sum(kron_power(exp_h(h, t), k) for h in generators)/24
            bound = t*t*(1-t*t*(1.5*k)**2/3)*a2
            check(np.linalg.eigvalsh(np.eye(6**k)-average.conj().T @ average-bound).min()
                  >= -2e-12, "quadratic contraction control")
            check(np.linalg.norm(average @ charge-charge @ average) < 1e-12,
                  "twirl preserves charge sector")
            value = np.linalg.norm(average @ charge, 2)
            check(value <= 1-delta*t*t/3+1e-12, "charged norm contraction")

    # General noncommuting zero-mean Hermitian ensembles.
    rng = np.random.default_rng(167)
    for _ in range(20):
        hs = []
        for _ in range(5):
            sample = rng.normal(size=(5, 5))+1j*rng.normal(size=(5, 5))
            hs.append((sample+sample.conj().T)/2)
        mean = sum(hs)/len(hs)
        hs = [h-mean for h in hs]
        hmax = max(np.linalg.norm(h, 2) for h in hs)
        a2 = sum(h @ h for h in hs)/len(hs)
        for bandwidth in [.01, .1, .5, 1]:
            t = bandwidth/hmax
            average = sum(exp_h(h, t) for h in hs)/len(hs)
            remainder = (np.eye(5)-average.conj().T @ average
                         -t*t*(1-t*t*hmax*hmax/3)*a2)
            check(np.linalg.eigvalsh(remainder).min() >= -1e-12,
                  "noncommuting mean-zero contraction")

    for n in [1, 10, 1e6, 1e12]:
        for leakage in [0, 1e-18, 1/n, 10]:
            t0 = min(n**(1/3), math.inf if leakage == 0 else leakage**(-1/3))
            beta = min(1, (c0/2)**(1/3))
            check((beta*t0)**3*leakage <= c0/2*(1+1e-12), "cubic cost flux branch")
            check(n/t0**2 >= t0*(1-1e-12), "cubic cost kinetic branch")
    print(f'{COUNT} checks for Theorem R.1 passed (finite controls, not a proof).')


if __name__ == '__main__':
    main()
