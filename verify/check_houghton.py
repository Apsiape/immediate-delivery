"""Houghton's channel Phi_H (Johnson's presentation; d = 104, 26 labels, Section 2.2). In its corner models the relators close
except r4 at one clock, and on the triangle-good, collision-free points the gadget unitary is an exact rectangular dilation with
exact label traces, log2 k of order M and room of order M^-2 (the corner families used in Appendix N.1), by a union bound.
Run: python -u check_houghton.py   (about 10 s)
"""
import os
import sys
import json
import time
from math import log2
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corner_tools.model import Model, REL, comp, invp, idx
from corner_tools import gadget as gd

t0 = time.time()
results = {}
ENT = gd.ENT
labels = {}
for e in ENT:
    labels.setdefault(gd.label(e[2]), []).append(e[2])
canon = {g: min(ws, key=lambda w: (len(w), w)) for g, ws in labels.items()}
WORDS = sorted(set(e[2] for e in ENT) | set(canon.values()))
print(f"Phi_H gadget: {len(ENT)} slots, {len(labels)} labels, d = {max(e[0] for e in ENT) + 1}")
results["slots"], results["labels"], results["d"] = len(ENT), len(labels), max(e[0] for e in ENT) + 1


def clock_data(model, i, j):
    paths = {w: model.word(i, j, gd.inverse(w)) for w in WORDS}
    clean = True
    for e in ENT:
        pa, pc = paths[e[2]], paths[canon[gd.label(e[2])]]
        if pa[:2] != pc[:2]:
            raise RuntimeError("clock mismatch: labels must fix translation")
        if comp(pa[2], invp(pc[2])) != model.I:
            clean = False
    seps = []
    gl = list(labels)
    for a in range(len(gl)):
        for b in range(a + 1, len(gl)):
            pg, ph = paths[canon[gl[a]]], paths[canon[gl[b]]]
            if pg[:2] != ph[:2]:
                continue
            sig = comp(pg[2], invp(ph[2]))
            if sig == model.I:
                raise RuntimeError("two labels collide for every colouring")
            seps.append(sig)
    return clean, seps


# ---------------------------------------------------------------- relator holonomies of the corner models
sanity_ok = True
for M in [5, 8, 12, 16]:
    m = Model(M)
    fails = {}
    for r in REL:
        for i in range(M):
            for j in range(M):
                ii, jj, p = m.word(i, j, r)
                if (ii, jj) != (i, j):
                    raise RuntimeError('relator does not close')
                if p != m.I:
                    fails.setdefault(r, []).append(((i, j), idx(p)))
    ok = (list(fails) == ['abABX'] and fails['abABX'] == [((M - 1, M - 1), M)])
    sanity_ok = sanity_ok and ok
    print(f"M={M}: failing (relator, clock, rank of colour permutation): {fails}")
results["sanity_ok"] = float(sanity_ok)

# ---------------------------------------------------------------- dilations, union bound
rows, dirty_ok, hist_ok = [], True, True
for M in [5, 6, 8, 10, 12, 16, 20, 24]:
    model = Model(M)
    L = model.L
    dirty, clean_ranks = [], []
    for i in range(M):
        for j in range(M):
            clean, seps = clock_data(model, i, j)
            if not clean:
                dirty.append((i, j)); continue
            clean_ranks.append([idx(sg) for sg in seps])
    dirty_ok = dirty_ok and dirty == [(M - 1, M - 1)] and all(len(cr) == 65 and min(cr) >= 1 for cr in clean_ranks)
    hist = Counter(tuple(sorted(Counter(cr).items())) for cr in clean_ranks).most_common(1)[0][0]
    if M >= 8:
        hist_ok = hist_ok and hist == ((1, 30), (2, 27), (3, 8))
    s = 64 * M * M
    good_frac = sum(max(0.0, 1 - sum(float(s) ** (-r) for r in cr)) for cr in clean_ranks) / (M * M)
    delta = -log2(good_frac)
    logk = 2 * log2(M) + L * log2(s) + log2(good_frac)
    rows.append((M, logk, delta))
    print(f"M={M:2d}: dirty clocks {dirty}; per clean clock 65 separators, most common rank histogram {dict(hist)}; "
          f"s=64M^2: log2 k >= {logk:8.2f}, log2(k'/k) <= {delta:.6f}, M^2 * room = {delta*M*M:.3f}")
results["dirty_ok"] = float(dirty_ok)
results["hist_ok"] = float(hist_ok)
results["room_M5"] = rows[0][2]
results["room_M24"] = rows[-1][2]
results["logk_M5"] = rows[0][1]
results["logk_M24"] = rows[-1][1]
results["room_times_M2"] = [r_[2] * r_[0] ** 2 for r_ in rows]
results["logk_over_M_M24"] = rows[-1][1] / 24
print(f"(t={time.time()-t0:.1f}s)")
print("RESULTS_JSON " + json.dumps(results, default=lambda o: o.item() if hasattr(o, "item") else float(o)))
