"""Cocycles of the right regular action of Houghton's group H_3 (Lemma M.1(b) with q = 3): for each of the 26 labels of Phi_H and
every translation (m, n) in [3, 40]^2, the cocycle of x -> x g^-1 is a finitary permutation of ray-1 points with index at most
m + n + 4, so it depends only on (m, n) and the label and lies in a finite symmetric group near the origin.
Run: python -u check_cocycles.py   (a few seconds)
"""
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corner_tools import gadget as gd

W = 400
NPTS = 3 * W
pid = lambda r, j: (r - 1) * W + (j - 1)          # point (r, j) -> index


def letter_array(c):
    P = np.arange(NPTS)
    for r in (1, 2, 3):
        for j in range(1, W + 1):
            pt = (r, j)
            if c in 'xX':
                img = {(1, 1): (1, 2), (1, 2): (1, 1)}.get(pt, pt)
            elif c == 'a':
                img = ((2, 1) if j == 1 else (1, j - 1)) if r == 1 else ((2, j + 1) if r == 2 else pt)
            elif c == 'A':
                img = ((1, 1) if j == 1 else (2, j - 1)) if r == 2 else ((1, j + 1) if r == 1 else pt)
            elif c == 'b':
                img = ((3, 1) if j == 1 else (1, j - 1)) if r == 1 else ((3, j + 1) if r == 3 else pt)
            else:  # 'B'
                img = ((1, 1) if j == 1 else (3, j - 1)) if r == 3 else ((1, j + 1) if r == 1 else pt)
            if img[1] <= W:
                P[pid(r, j)] = pid(*img)
    return P


LET = {c: letter_array(c) for c in 'aAbBxX'}
INV = dict(zip('aAbBxX', 'AaBbXx'))


def word_array(w):
    idx = np.arange(NPTS)
    for c in w:          # right action: apply the letters in reading order
        idx = LET[c][idx]
    return idx


def compose(*arrs):      # first arrs[0], then arrs[1], ...
    idx = np.arange(NPTS)
    for P in arrs:
        idx = P[idx]
    return idx


def power(c, k):
    return word_array(c * k)


PA = {k: power('a', k) for k in range(0, 46)}
PAi = {k: power('A', k) for k in range(0, 46)}
PB = {k: power('b', k) for k in range(0, 46)}
PBi = {k: power('B', k) for k in range(0, 46)}


def t_arr(m, n):          # word a^m b^n (m, n >= 0)
    return compose(PA[m], PB[n])


def tinv_arr(m, n):       # word (a^m b^n)^-1 = B^n A^m
    return compose(PBi[n], PAi[m])


mass, words = {}, {}
for e in gd.ENT:
    g = gd.label(e[2])
    mass[g] = mass.get(g, 0.0) + 1.0 / e[4]
    words.setdefault(g, []).append(e[2])
canon = {g: min(ws, key=lambda w: (len(w), w)) for g, ws in words.items()}
d = 104
check_pts = np.array([pid(r, j) for r in (1, 2, 3) for j in range(1, 61)])
ray_of = check_pts // W + 1
j_of = check_pts % W + 1
Kmax_all, all_ray1, Ks = -99, True, []
for g, w in sorted(canon.items(), key=lambda kv: (kv[0][0], kv[1])):
    (u, v), _ = g
    winv = word_array(''.join(INV[c] for c in reversed(w)))
    Kg = -99
    for m in range(3, 41):
        for n in range(3, 41):
            c = compose(t_arr(m, n), winv, tinv_arr(m - u, n - v))
            moved = c[check_pts] != check_pts
            if moved.any():
                if np.any(ray_of[moved] != 1):
                    all_ray1 = False
                Kg = max(Kg, int(j_of[moved].max()) - (m + n))
    Ks.append(Kg)
    Kmax_all = max(Kmax_all, Kg)
step = {}
for g in canon:
    (u, v), _ = g
    step[(-u, -v)] = step.get((-u, -v), 0.0) + mass[g] / d
res = dict(labels=len(canon), max_K=Kmax_all, all_on_ray1=1.0 if all_ray1 else 0.0,
           sum_mu=sum(mass.values()) / d, step_keys=[list(k) for k in sorted(step)],
           step_law=[round(step[k], 5) for k in sorted(step)])
print(f"labels {len(canon)}, sum mu = {res['sum_mu']:.6f}; cocycles move only ray-1 points: {all_ray1}; "
      f"largest moved index - (m + n) over all labels and (m, n) in [3,40]^2: {Kmax_all}")
print("step law of x -> x g^-1:", {k: round(val, 5) for k, val in sorted(step.items())})
print("RESULTS_JSON " + json.dumps(res))
