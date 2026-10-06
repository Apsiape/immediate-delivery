"""Permutation rounds. Part A: Lemma 2.3 on an S_3 gadget (d = 36): a pure bath point leaks nothing if it is triangle-good,
even with label collisions, and at least 1/(4d) otherwise. Parts B, C: on the Z gadget, gluing the level sets of a Gaussian
bath into a flat bath pays room of order sqrt(a), against order a for flat intervals, before and after smoothing; this
illustrates the remark after Theorem H.12.
Run: python -u check_permutation.py
"""
import itertools
import json
import numpy as np

results = {}

# ---------------------------------------------------------------- Part A
print("=== Part A: leakage at pure bath points of a permutation model of the S_3 gadget ===")
rng = np.random.default_rng(11)


def mul(p, q):
    return tuple(p[q[i]] for i in range(3))


def inv(p):
    r = [0] * 3
    for i, pi in enumerate(p):
        r[pi] = i
    return tuple(r)


E = (0, 1, 2)
A = (1, 0, 2)
B = (0, 2, 1)
J = mul(A, B)
gens = {'a': A, 'b': B, 'j': J}


def ev(word):
    g = E
    for (x, e) in word:
        g = mul(g, gens[x] if e == 1 else inv(gens[x]))
    return g


R = [[('a', 1), ('a', 1)], [('b', 1), ('b', 1)], [('a', 1), ('b', 1)] * 3, [('j', -1), ('a', 1), ('b', 1)]]
assert all(ev(r) == E for r in R)
letters = [(x, e) for x in 'abj' for e in (1, -1)]
symbols = [tuple([l]) for l in letters]
triangles = []
for r in R:
    L = len(r)
    pref = {1: tuple(r[:1])}
    for l in range(2, L):
        pref[l] = tuple(r[:l])
        symbols.append(pref[l])
    pref[L] = ()
    for l in range(2, L + 1):
        triangles.append((pref[l - 1], tuple([r[l - 1]]), pref[l]))
for x in 'abj':
    triangles.append((((x, 1),), ((x, -1),), ()))
d = 1 + len(symbols) + 2 * len(triangles)
slots = [(0, 0, 1.0, 'word', ())]
for i, w in enumerate(symbols):
    slots.append((1 + i, 1 + i, 1.0, 'word', w))
base = 1 + len(symbols)
h = 1 / np.sqrt(2)
for t, (x, y, z) in enumerate(triangles):
    r0 = base + 2 * t
    slots += [(r0, r0, h, 'word', ()), (r0, r0 + 1, h, 'word', y), (r0 + 1, r0, h, 'word', x),
              (r0 + 1, r0 + 1, -h, 'pair', (x, y))]
labels = sorted(set(ev(w) for w in [()] + symbols))


def slot_label(s):
    if s[3] == 'word':
        return ev(s[4])
    x, y = s[4]
    return mul(ev(x), ev(y))


Amat = []
for g in labels:
    Am = np.zeros((d, d))
    for s in slots:
        if slot_label(s) == g:
            Am[s[0], s[1]] += s[2]
    Amat.append(Am.reshape(-1))
Qb, _ = np.linalg.qr(np.array(Amat).T)
Pi = Qb @ Qb.T
els = list(itertools.permutations(range(3)))
m = 20
Om = [(g, i) for g in els for i in range(m)]
idx = {w: n for n, w in enumerate(Om)}
M = len(Om)
f = {g: np.array([idx[(mul(g, w[0]), w[1])] for w in Om]) for g in labels}
pts = rng.choice(M, 5, replace=False)
pert = np.arange(M)
pert[pts] = pts[rng.permutation(5)]
f[J] = pert[f[J]]


def pi_of(s):
    if s[3] == 'word':
        return f[ev(s[4])]
    x, y = s[4]
    return f[ev(x)][f[ev(y)]]


def leak(w):
    groups = {}
    for s in slots:
        p = pi_of(s)[w]
        groups.setdefault(p, np.zeros((d, d)))
        groups[p][s[0], s[1]] += s[2]
    tot = 0.0
    for K in groups.values():
        v = K.reshape(-1)
        tot += v @ v - v @ (Pi @ v)
    return tot / d


def tri_good(w):
    return all(f[ev(x)][f[ev(y)][w]] == f[ev(z)][w] for (x, y, z) in triangles)


def collides(w):
    imgs = [f[g][w] for g in labels]
    return len(set(imgs)) < len(imgs)


good, bad = [], []
for w in range(M):
    (good if tri_good(w) else bad).append(leak(w))
print(f"d = {d}, labels = {len(labels)}, M = {M}: good points {len(good)}, bad points {len(bad)}")
print(f"max leakage at good points = {max(good):.2e}; min leakage at bad points = {min(bad):.6f} "
      f"(= {min(bad) * d:.3f}/d; bound 1/(4d) = {1 / (4 * d):.6f})")
results["perm_d"] = d
results["perm_good_points"], results["perm_bad_points"] = len(good), len(bad)
results["perm_good_max_leak"] = max(good)
results["perm_bad_min_leak_times_d"] = min(bad) * d
cos = list(dict.fromkeys(frozenset([g, mul(g, A)]) for g in els))
cidx = {c: n for n, c in enumerate(cos)}
f = {g: np.array([cidx[frozenset(mul(g, x) for x in c)] for c in cos]) for g in labels}
lk = [leak(w) for w in range(len(cos))]
ncoll = sum(collides(w) for w in range(len(cos)))
print(f"exact non-free action on S_3/<a>: colliding points {ncoll} of {len(cos)}, max leakage {max(lk):.2e}")
results["perm_collision_points"] = ncoll
results["perm_collision_max_leak"] = max(lk)

# ---------------------------------------------------------------- Part B
print("\n=== Part B: glued level sets of exact Gaussian rounds on the Z gadget ===")
mu = np.array([0.3, 0.4, 0.3])


def H(q):
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


gauss = {}
for w in [2, 4, 8, 16, 32, 64, 128, 256]:
    x = np.arange(-12 * w, 12 * w + 1)
    p = np.exp(-x ** 2 / (2.0 * w * w)); p /= p.sum()
    a = H(np.convolve(p, mu)) - H(p)
    room = np.log2(1 + 2 * p.max())
    gauss[w] = room / np.sqrt(a)
    print(f"w={w:4d}: S={H(p):8.4f} a={a:.3e} room={room:.4e} room/sqrt(a)={room/np.sqrt(a):.4f} room/a={room/a:8.2f}")
flat = {}
for l in [4, 16, 64, 256, 1024]:
    p = np.ones(l) / l
    a = H(np.convolve(p, mu)) - H(p)
    flat[l] = np.log2(1 + 2.0 / l) / a
    print(f"flat interval l={l:5d}: a={a:.3e} room/a={flat[l]:.4f}")
results["levelset_room_over_sqrt_a_w256"] = gauss[256]
results["levelset_room_over_sqrt_a_w8"] = gauss[8]
results["flat_room_over_a_l1024"] = flat[1024]

# ---------------------------------------------------------------- Part C
print("\n=== Part C: smoothed room of the Gaussian rounds (eps = 1/16, eta = eps/n) ===")
eps = 1 / 16
smooth = {}
for w in [8, 32, 128, 512]:
    x = np.arange(-40 * w, 40 * w + 1)
    p = np.exp(-x ** 2 / (2.0 * w * w)); p /= p.sum()
    a = H(np.convolve(p, mu)) - H(p)
    row = []
    for e in (4, 8, 12):
        eta = eps / 10 ** e
        order = np.argsort(-p)
        mm = int(np.searchsorted(np.cumsum(p[order]), 1 - eta)) + 1
        room = np.log2((mm + 2) / mm)
        row.append(room / np.sqrt(a))
    smooth[w] = row
    print(f"w={w:4d}: a={a:.3e}  room_eta/sqrt(a) at n = 1e4, 1e8, 1e12: " + ", ".join(f"{v:.3f}" for v in row))
results["smoothed_w8"] = smooth[8]
results["smoothed_w512"] = smooth[512]
print("RESULTS_JSON " + json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
