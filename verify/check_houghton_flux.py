"""Finite controls for Theorem Q.4 (fourth-power leakage bound): wedge count and density >= 0.45, area residues mod 4, the
stabilized logical-point geometry, the fixed five-point gates (trace, minimal polynomial z^2 - z - 7/8, dense in SU(4)),
cancellation of outside phases, and the final two-branch cost algebra. Consistency checks, not a proof.
"""
from pathlib import Path
import math
import numpy as np
from check_houghton_entropy import root, state


COUNT = 0


def check(condition, label):
    global COUNT
    COUNT += 1
    if not condition:
        raise AssertionError(label)


def phase(dim, v, angle):
    return np.eye(dim) + (np.exp(1j*angle)-1)*np.outer(v, v.conj())


def rotate(v, x, y, angle):
    z = (np.exp(1j*angle)-1)*(v[x]-v[y])/2
    v[x] += z
    v[y] -= z


def geometry(rng, m, i0, j0):
    """Actual inverse-prefix action, sampled sparse eligible potentials.

    Constant-curl potential values are unnecessary for these exact
    support and stable-basis identities. Field areas are checked below.
    """
    x0, dim = m//4, 4*m+4
    r = x0+8
    check(0 <= i0 <= m-x0-12 and 1 <= j0 <= m-3
          and i0+j0 >= m-x0+16, "corrected good wedge")
    steps, totals = [], {}
    for q in range(r-1):
        i = i0+q
        check(i <= m-6, "no seam or preliminary A move")
        rotations = []
        for t in range(4, m+4):
            if i+j0 > (m-3 if t == 4 else t+m-8):
                continue
            x, y = t+m-1-i-j0, t+2*m-1-j0
            s = x+q
            check(y == s+m+i0, "fixed pulled-back pair")
            check(q <= s-(6 if t == 4 else 7), "stabilized logical point")
            if rng.random() < .5:
                continue
            angle = float(rng.uniform(-math.pi, math.pi))
            rotations.append((x-1, y-1, angle))
            totals[s] = totals.get(s, 0)+angle
        steps.append((2*m+2-i-j0, rotations))
    check(steps[-1][0] >= 11, "last base-cycle margin")
    logical = []
    for s in range(1, r+2):
        v = np.zeros(dim, complex)
        v[s-1] = 1
        if s in totals:
            rotate(v, s-1, s+m+i0-1, -totals[s])
        logical.append(v)
    f = np.array(logical)
    check(np.linalg.norm(f.conj() @ f.T-np.eye(r+1)) < 1e-12,
          "orthonormal logical point family")
    check(np.linalg.norm(f[:3]-np.eye(dim)[:3]) < 1e-12,
          "first three logical points unchanged")
    for j in range(r):
        v = np.zeros(dim, complex)
        v[0], v[1] = 1/math.sqrt(2), -1/math.sqrt(2)
        for n, rotations in reversed(steps[:j]):
            v[:n] = np.roll(v[:n], 1)
            for x, y, angle in rotations:
                rotate(v, x, y, -angle)
        target = (f[j]-f[j+1])/math.sqrt(2)
        check(np.linalg.norm(v-target) < 2e-12, "common D gives exact normals")

    selected = []
    for r0 in range(8):
        s = x0+r0
        t = s-m+1+i0+j0
        h = m+3-t
        area = (m-1)**2-h*(h+1)//2
        check(17 <= t <= m-7, "selected region in untruncated area range")
        check(i0+j0 <= t+m-8, "selected bulk region active")
        check(t+m-1-i0-j0 == s, "selected field on logical point")
        if area % 4 == 0:
            selected.append((s, area))
    check(bool(selected), "eight consecutive areas contain divisible-by-four")
    for s, area in selected:
        check(area/4 <= m*m/4, "selected power bound")
        check(4*s <= m+28, "collapsed swap length")


def main():
    rng = np.random.default_rng(166)
    for m in (4096, 4097, 8192, 10000):
        x = m//4
        count = sum(m-3-(m-x+16-i)+1 for i in range(m-x-11))
        formula = (m-x-11)*(m+x-48)//2
        check(count == formula, "corrected wedge exact count")
        check(count/m**2 >= 15/32-40/m >= .45, "good-clock density")
    for residue in range(8):
        triangular = [(h*(h+1)//2) % 4 for h in range(residue, residue+8)]
        check(0 in triangular and 1 in triangular, "all area square residues")
    for clock in ((0, 125), (84, 125), (84, 48), (32, 125)):
        geometry(rng, 128, *clock)

    dim = 5
    plus = np.ones(dim)/math.sqrt(dim)
    v = []
    for i in range(4):
        z = np.zeros(dim)
        z[i], z[4] = 1/math.sqrt(2), -1/math.sqrt(2)
        v.append(z)
    g = [phase(dim, z, math.pi/4) for z in v]
    fixed = []
    expected_trace = (2+3*math.sqrt(2))/4
    for i in range(4):
        for j in range(i+1, 4):
            u = g[i] @ g[j].conj().T
            fixed.extend((u, g[i] @ u @ g[i].conj().T))
            check(abs(np.trace(u)-3-expected_trace) < 1e-12,
                  "fixed two-plane trace")
            check(np.linalg.norm(g[i] @ u-u @ g[i]) > .01,
                  "nonparallel conjugate axes")
    check(abs(((2+3*math.sqrt(2))/4)*((2-3*math.sqrt(2))/4)+7/8) < 1e-14,
          "nonintegral field norm")
    basis = []

    def add(a):
        a = a.copy()
        for b in basis:
            a -= np.vdot(b, a).real*b
        norm = np.linalg.norm(a)
        if norm > 1e-10:
            basis.append(a/norm)

    for u in fixed:
        check(np.linalg.norm(u @ plus-plus) < 1e-12, "fixed inert mode")
        check(abs(np.linalg.det(u)-1) < 1e-12, "determinant-one fixed gate")
        add(u-u.conj().T)
    for _ in range(4):
        old = list(basis)
        for a in old:
            for b in old:
                add(a @ b-b @ a)
        if len(basis) == len(old):
            break
    check(len(basis) == 15, "fixed Lie control spans su4")

    # A common outside phase cancels in the twelve local generators.
    outside = np.diag([1, 1, 1, 1, 1, np.exp(.31j), np.exp(-.19j)])
    w = np.eye(7, dtype=complex)
    w[:5, :5] = g[3]
    w = w @ outside
    controlled = []
    for i in range(4):
        c = np.eye(7)
        c[:, [i, 3]] = c[:, [3, i]]
        controlled.append(c @ w @ c.T)
    for i in range(4):
        for j in range(i+1, 4):
            u = controlled[i] @ controlled[j].conj().T
            check(np.linalg.norm(u[:5, :5]-g[i] @ g[j].conj().T) < 1e-12,
                  "local ratio exact")
            check(np.linalg.norm(u[5:, 5:]-np.eye(2)) < 1e-12,
                  "outside phases cancel")

    for p in (1, 2, 7, 32):
        rho = state(rng, 7)
        xi = root(rho)
        c, _ = np.linalg.qr(rng.normal(size=(7, 7))+1j*rng.normal(size=(7, 7)))
        base = np.diag(np.exp(1j*rng.uniform(-.1, .1, 7)))
        gp = c @ np.linalg.matrix_power(base, p) @ c.conj().T
        action = np.linalg.norm((gp-np.eye(7)) @ xi)
        e = np.linalg.norm((base-np.eye(7)) @ xi)
        centrality = np.linalg.norm(c @ xi @ c.conj().T-xi)
        check(action <= p*e+2*centrality+1e-12, "action/centrality comparison")

    # Corollary branch algebra, using illustrative positive constants only.
    cent, c0 = 1/(1280*2*math.sqrt(104*math.log(2))), 1e-8
    beta = min(1, (c0/2)**.25)
    for n in (1, 10, 1e6, 1e12):
        for leakage in (0, 1e-18, 1/n, n**(-4/3), 10):
            t0 = min(n**(1/3), math.inf if leakage == 0 else leakage**(-.25))
            check(t0**3 <= n*(1+1e-12), "two-scale minimum")
            check((beta*t0)**4*leakage <= c0/2*(1+1e-12), "flux branch")
            check(n/t0**2 >= t0*(1-1e-12), "kinetic branch")
            check(min(cent*beta, c0/(2*beta**2)) > 0, "uniform corollary constant")

    print(f'{COUNT} checks for Theorem Q.4 passed (finite controls, not a proof).')


if __name__ == '__main__':
    main()
