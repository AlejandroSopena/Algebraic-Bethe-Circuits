import numpy as np
from numpy.linalg import qr


def _r_matrix_xxz(root, delta):
    """
    R matrix used in the 6-vertex model to construct the state of interest. 

    Args:
        root (complex): root of the Bethe equations.
        delta (float): Anisotropy Delta. 

    Returns:
        r_matrix (array): R matrix.
    """
    r_matrix = np.eye(4, dtype=np.complex128)
    if delta == 1:
        b = (root - 1j) / (root + 1j)
        c = 2j / (root + 1j)
    elif delta > 1:
        gamma = np.arccosh(delta)
        b = np.sin(gamma / 2 * (root - 1j)) / np.sin(gamma / 2 * (root + 1j))
        c = 1j * np.sinh(gamma) / np.sin(gamma / 2 * (root + 1j))
    else:
        gamma = np.arccos(delta)
        b = np.sinh(gamma / 2 * (root - 1j)) / np.sinh(gamma / 2 * (root + 1j))
        c = 1j * np.sin(gamma) / np.sinh(gamma / 2 * (root + 1j))
    r_matrix[1, 1] = r_matrix[2, 2] = c
    r_matrix[1, 2] = r_matrix[2, 1] = b
    return r_matrix


def RT(roots, delta):
    """
    Basic cell R_T.

    Args:
        root (complex): root of the Bethe equations.
        delta (float): Anisotropy Delta.

    Returns:
        r_matrix (array): R matrix.
    """
    Id = np.identity(2)
    nmagnons = len(roots)
    Rs = []
    for k in range(nmagnons):
        Rs.append(_r_matrix_xxz(roots[k], delta))
    A = []
    for j in range(nmagnons):
        l = []
        for i in range(nmagnons):
            if j == i:
                l.append(Rs[j])
            else:
                l.append(Id)
        a = l[0]
        for k in range(1, nmagnons):
            a = np.kron(a, l[k])
        A.append(a)
    A = A[::-1]
    p = A[0]
    for k in range(1, nmagnons):
        p = p @ A[k]
    return p


def full_pink(nspins, roots, delta):
    """
    P_k and G_k matrices.

    Args:
        nspins (integer): number of sites.
        roots (array [nmagnons]): roots of the Bethe equations.
        delta (float): Anisotropy Delta.

    Returns:
        [P_k (list), G_k (list)]: P_k and G_k matrices.
    """
    nmagnons = len(roots)
    Rt = RT(roots, delta)
    G0 = RT(roots, delta)[0:2, 0:2 ** nmagnons]
    Gs = [G0]
    Ps = []
    Id = np.identity(2)
    for j in range(1, nmagnons):
        DB = (np.kron(Gs[j - 1], Id) @ Rt)[:, 0:2 ** nmagnons]
        Pj, Gj = qr(DB)
        Gs.append(Gj)
        Ps.append(Pj)
    for j in range(nmagnons, nspins):
        DB = (np.kron(Gs[j - 1], Id) @ Rt)[:, 0:2 ** nmagnons]
        Pj, Gj = qr(DB)
        Pj1 = np.zeros((2 ** (nmagnons + 1), 2 ** (nmagnons + 1)), complex)
        Pj1[:, 0:2 ** nmagnons] = Pj
        Gs.append(Gj)
        Ps.append(Pj1)
    return [Ps, Gs]
