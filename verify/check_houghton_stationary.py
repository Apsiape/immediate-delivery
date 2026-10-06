"""Finite controls for Corollary Q.5 (stationary sources): b = pi^2/(2048 * 8192), beta = b/80 and 9/16 - 11/20 = 1/80, the
selected rational powers and pair counts, generation of SU(H) by the local groups, and a qubit example where small gain does not
give proximity to the stationary state (remark after Corollary Q.5). Not a proof.
"""
import math
import numpy as np


def main():
    count = 0

    def check(condition, label):
        nonlocal count
        count += 1
        if not condition:
            raise AssertionError(label)

    q0, q1, loss = 7/12, 9/16, 11/20
    b = math.pi**2/(2048*8192)
    beta = b/80  # The finite-clock minimum can only reduce this value.
    eps = min((math.sqrt(q0)-math.sqrt(q1))**2/4, beta*q1/16)
    check(abs(q1-loss-1/80) < 1e-15, "stationary threshold charge")
    check(abs(13/30-(1-q0)-1/60) < 1e-15, "global sufficient gap slack")
    check(9/20 > 13/30, "kernel sector has strictly larger gap")
    check(math.sqrt(q0)-math.sqrt(eps) >= math.sqrt(q1),
          "root projection retains charge")
    check((math.sqrt(beta*q1)-math.sqrt(eps))**2 > beta*q1/4,
          "observable criterion retains capped flux")
    check(abs((q0-q1)/(1-q1)-1/21) < 1e-15, "zero-gain flag weight")
    x = 9/16384
    f6 = -x*math.log2(x)-(1-x)*math.log2(1-x)+x*math.log2(5)
    check(q1-7/128-f6 > 7/16, "same Theorem Q.3 constant at weaker charge")
    check(9*math.pi**2/2048 < 1, "selected phases fit the natural cap")

    for m in (4096, 4097, 8192, 10001, 65536):
        x, z = m//4, m//256
        start = x-z + (x-z) % 2
        number = (z-1)//8
        check(m/8192 <= number <= m/2048, "selected pair count")
        check(start+8*(number-1)+7 <= x, "selected pairs inside chain")
        for i in (0, (m-x-12)//2, m-x-12):
            lower = m-x+16+2*z-i
            for clock_j in (lower, m-3):
                for pair_j in (0, number-1):
                    candidates = []
                    for s in range(start+8*pair_j, start+8*pair_j+8):
                        t = s-m+1+i+clock_j
                        h = m+3-t
                        area = (m-1)**2-h*(h+1)//2
                        check(5 <= t <= m+3 and i+clock_j <= t+m-8,
                              "actual bulk area selection")
                        check(m*m/3 <= area <= m*m, "actual area bounds")
                        if area % 4 == 0:
                            candidates.append(s)
                    check(bool(candidates), "eight-index rational power exists")

    # Small instances of the constructive Lie argument; no enumeration of
    # exterior flavour spaces or all representations is attempted.
    for pairs in (1, 2, 3):
        n = 3+2*pairs
        e = np.zeros(n)
        e[0], e[1] = 1/math.sqrt(2), -1/math.sqrt(2)
        frames = []
        for j in range(pairs):
            indices = [0, 1, 2, 3+2*j, 4+2*j]
            embed = np.eye(n)[:, indices]
            local = embed @ (np.eye(5)-np.ones((5, 5))/5) @ embed.T
            check(np.linalg.norm(local @ e-e) < 1e-13, "common anchor plane")
            frames.append(local-np.outer(e, e))
        span = np.concatenate(frames, axis=1)
        u, values, _ = np.linalg.svd(span, full_matrices=False)
        dimension = int(np.count_nonzero(values > 1e-10))
        check(dimension == n-2, "local perpendicular spaces span H perpendicular")
        modes = [e] + [u[:, j] for j in range(dimension)]
        generators = []
        for v in modes[1:]:
            generators.extend((np.outer(e, v)-np.outer(v, e),
                               1j*(np.outer(e, v)+np.outer(v, e))))
        commutators = [a @ c-c @ a for a in generators for c in generators]
        columns = np.array([np.concatenate((a.real.ravel(), a.imag.ravel()))
                            for a in generators+commutators]).T
        check(np.linalg.matrix_rank(columns, tol=1e-10) == (n-1)**2-1,
              "star generators and commutators span su(H)")

    # Small actual gain need not imply Haar-stationary trace proximity.
    rho = np.diag([1., 0.])
    pauli_x = np.array([[0., 1.], [1., 0.]])
    for q in (3, 7, 101):
        theta = math.pi/q
        v = math.cos(theta)*np.eye(2)+1j*math.sin(theta)*pauli_x
        output = (rho+v @ rho @ v.conj().T)/2
        values = np.linalg.eigvalsh(output)
        expected = (1-math.cos(theta))/2
        check(abs(values[0]-expected) < 1e-13, "qubit gain eigenvalues")
        eta = sum(np.linalg.matrix_power(v, j) @ rho
                  @ np.linalg.matrix_power(v, j).conj().T for j in range(q))/q
        check(np.linalg.norm(eta-np.eye(2)/2) < 1e-12, "actual stationary average")
        check(abs(np.linalg.svd(rho-eta, compute_uv=False).sum()/2-.5) < 1e-12,
              "stationary distance remains one half")
    print(f"{count} checks for Corollary Q.5 passed (finite controls, not a proof).")


if __name__ == "__main__":
    main()
