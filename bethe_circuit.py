import qibo

qibo.set_backend("qibojit")
qibo.set_device("/CPU:0")
import numpy as np
from numpy.linalg import norm
from qibo import gates
from qibo.models import Circuit
import Pk_gates as P

np.seterr(invalid='raise')


class BetheCircuit:
    """
    Class for quantum circuits implementing the Algebraic Bethe Ansatz.
    """

    def __init__(self, nspins, nmagnons, delta):
        """
        Args:
            nspins (int): number of sites.
            nmagnons (int): number of magnons.
        """
        self.nspins = nspins
        self.nmagnons = nmagnons
        self.delta = delta

    def aba(self, roots):
        """
        Circuit built from the matrices P_k. 
        
        Args:
            roots (array [nmagnons,1]): roots of the Bethe equations.
        
        Returns:
            circuit (circuit): quantum circuit.
        """
        self.circuit = Circuit(self.nspins)
        unitaries = self._unitary_gates(roots)
        for n in range(self.nmagnons):
            self.circuit.add(gates.X(n))
        for n in range(self.nspins - self.nmagnons):
            qubits = list(np.arange(n, n + self.nmagnons + 1, 1))[::-1]
            self.circuit.add(gates.Unitary(unitaries[n], *qubits))
        j = 0
        for n in range(self.nspins - self.nmagnons, self.nspins - 1):
            qubits = list(np.arange(self.nspins - self.nmagnons + j, self.nspins, 1))[::-1]
            self.circuit.add(gates.Unitary(unitaries[n], *qubits))
            j += 1
        return self.circuit

    def _unitary_gates(self, roots):
        """
        P_k matrices. Unitaries P_k for k<M and isometries P_k|0> for k>=M. 
        
        Args:
            roots (array [nmagnons]): roots of the Bethe equations.
        
        Returns:
            unitary_list (list [nspins-1]): P_k matrices. 
        """
        Pg = P.full_pink(self.nspins, roots, self.delta)[0]
        unitary_list = Pg
        return (unitary_list[::-1])

    def exact(self, roots):
        """
        Circuit built from the non-unitary R matrices. 
        
        Args:
            roots (array [nmagnons]): roots of the Bethe equations.
        
        Returns:
            tensornet (circuit): non unitary circuit.
        """
        self.tensornet = Circuit(self.nspins + self.nmagnons)
        for n in range(self.nmagnons):
            self.tensornet.add(gates.X(n))
        for i, m in enumerate(range(self.nmagnons, 0, -1)):
            for n in range(self.nspins):
                self.tensornet.add(gates.Unitary(self._r_matrix_xxz(roots[i]), n + m - 1, n + m))
        return self.tensornet

    def _r_matrix_xxz(self, root):
        """
        R matrix used in the 6-vertex model to construct the state of interest. 
        
        Args:
            root (complex): root of the Bethe equations.
        
        Returns:
            r_matrix (array): R matrix.
        """
        r_matrix = np.eye(4, dtype=np.complex128)
        if self.delta == 1:
            b = (root - 1j) / (root + 1j)
            c = 2j / (root + 1j)

        elif self.delta > 1:
            gamma = np.arccosh(self.delta)
            b = np.sin(gamma / 2 * (root - 1j)) / np.sin(gamma / 2 * (root + 1j))
            c = 1j * np.sinh(gamma) / np.sin(gamma / 2 * (root + 1j))
        else:
            gamma = np.arccos(self.delta)
            b = np.sinh(gamma / 2 * (root - 1j)) / np.sinh(gamma / 2 * (root + 1j))
            c = 1j * np.sin(gamma) / np.sinh(gamma / 2 * (root + 1j))
        r_matrix[1, 1] = r_matrix[2, 2] = c
        r_matrix[1, 2] = r_matrix[2, 1] = b
        return r_matrix
