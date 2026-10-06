"""Canonical gadget channels (Definition 2.2, equation (1)) for relators written in lower/upper-case letters (upper = inverse).
A slot is (row, col, coefficient, literal word); its label is the group element given by label_of(word). Letter operators come from
a model; a word's operator is the product left to right. Equal prefixes are listed once; tag them as in Definition 2.2 if relators
share prefixes.
"""
import numpy as np

S2 = 1 / np.sqrt(2)


def inv_letter(c):
    return c.lower() if c.isupper() else c.upper()


def build_gadget(gens, relators):
    symbols = []
    for x in gens:
        symbols += [x, x.upper()]
    for r in relators:
        for l in range(2, len(r)):
            if r[:l] not in symbols:
                symbols.append(r[:l])
    triangles = []
    for r in relators:
        L = len(r)
        for l in range(2, L + 1):
            x = r[0] if l == 2 else r[:l - 1]
            y = r[l - 1]
            z = r[:l] if l < L else ''
            triangles.append((x, y, z))
    for x in gens:
        triangles.append((x, x.upper(), ''))
    slots = [(0, 0, 1.0, '')]
    pos = 1
    for w in symbols:
        slots.append((pos, pos, 1.0, w)); pos += 1
    for (x, y, z) in triangles:
        slots += [(pos, pos, S2, ''), (pos, pos + 1, S2, y), (pos + 1, pos, S2, x), (pos + 1, pos + 1, -S2, x + y)]
        pos += 2
    d = pos
    return dict(symbols=symbols, triangles=triangles, slots=slots, d=d)


def word_op(word, letter_ops, dim):
    M = np.eye(dim, dtype=complex)
    for c in word:
        M = M @ letter_ops[c]
    return M


def z2_label(word):
    u = sum(1 if c == 'a' else -1 if c == 'A' else 0 for c in word)
    v = sum(1 if c == 'b' else -1 if c == 'B' else 0 for c in word)
    return (u, v)


def twisted_phase(word, omega):
    """phase zeta with lambda(word) = zeta * lambda(a)^u lambda(b)^v in the twisted regular rep on l2(Z^2),
    lambda(a) d_(i,j) = d_(i+1,j), lambda(b) d_(i,j) = omega^i d_(i,j+1). Apply letters right to left to d_(0,0)."""
    i, j, ph = 0, 0, 1.0 + 0j
    for c in reversed(word):
        if c == 'a':
            i += 1
        elif c == 'A':
            i -= 1
        elif c == 'b':
            ph *= omega ** i; j += 1
        elif c == 'B':
            j -= 1; ph *= np.conj(omega ** i)
    # section lambda(a)^u lambda(b)^v applied to d_(0,0): lambda(b)^v gives d_(0,v) with phase 1, then shift: phase 1
    return ph


def kraus_from_gadget(G, label_of, phase_of):
    d = G['d']
    labels = []
    for s in G['slots']:
        g = label_of(s[3])
        if g not in labels:
            labels.append(g)
    A = {g: np.zeros((d, d), complex) for g in labels}
    for (row, col, c, w) in G['slots']:
        A[label_of(w)][row, col] += c * phase_of(w)
    return labels, A


def choi_from_slot_gram(G, Gram):
    """normalized Choi of X -> sum_{a,b} Gram[a,b] c_a conj(c_b) E_a X E_b^*  (Gram[a,b] = tau(M_b^* M_a))."""
    d = G['d']
    vecs = []
    for (row, col, c, w) in G['slots']:
        v = np.zeros(d * d, complex)
        v[row * d + col] = c        # vec(E_a) = |row>|col>
        vecs.append(v)
    V = np.array(vecs).T             # (d^2, #slots)
    return V @ Gram @ V.conj().T / d


def choi_from_kraus(Ks, d):
    J = np.zeros((d * d, d * d), complex)
    for K in Ks:
        v = K.reshape(-1)            # vec with row-major: |row>|col>
        J += np.outer(v, v.conj())
    return J / d
