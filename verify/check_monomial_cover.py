"""Control for the d-fold cyclic cover in the proof of Theorem D.1: pairing two parallel identity edges at one bath pair gives
the identity channel, while the cover with distinct shifts gives the correct dephasing channel. Not a proof.
"""
import numpy as np


def choi(unitary, d, m):
    u = unitary.reshape(d, m, d, m)
    result = np.zeros((d*d, d*d), dtype=complex)
    for output_bath in range(m):
        for input_bath in range(m):
            vector = u[:, output_bath, :, input_bath].reshape(-1)
            result += np.outer(vector, vector.conj()) / (d*m)
    return result


def run():
    # One input/output tile has two singleton identity edges. Direct pairing
    # at the same bath pair merges the edges and incorrectly gives identity.
    # The two-fold cyclic cover assigns distinct shifts and gives dephasing.
    identity = np.eye(2)
    cover = np.zeros((4, 4), dtype=complex)
    for system in range(2):
        for bath in range(2):
            cover[2*system + ((bath+system) % 2), 2*system+bath] = 1
    target = np.diag([0.5, 0, 0, 0.5])
    naive = choi(identity, 2, 1)
    correct = choi(cover, 2, 2)
    assert np.allclose(cover.conj().T @ cover, np.eye(4))
    assert np.allclose(correct, target)
    assert not np.allclose(naive, target)
    print("parallel-edge cyclic-cover control passed")
    print("covered Choi residual", float(np.linalg.norm(correct-target)))
    print("naive Choi residual", float(np.linalg.norm(naive-target)))


if __name__ == "__main__":
    run()
