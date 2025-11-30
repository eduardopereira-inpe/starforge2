import numpy as np
from itertools import islice, count

class TikhonovOrder2(object):
    """
    Tikhonov regularization of order 2 (second difference operator).

    Attributes:
        _Nx (int): Size of the input vector.
        _Omega (np.ndarray): Second difference matrix.
        name (str): Name of the regularization.
        type (str): Type identifier.
    """

    def __init__(self, Nx):
        """
        Args:
            Nx (int): Size of the input vector.
        """
        self._Nx = Nx
        self._Omega = np.zeros((self._Nx, self._Nx))
        self.name = 'Tikhonov Order 2'
        self.type = 'tikhonovorder2'

        for i in islice(count(), 1, self._Nx-1):
            self._Omega[i][i] = -2.
            if i + 1 < Nx:
                self._Omega[i][i+1] = 1.
            if i - 1 >= 0:
                self._Omega[i][i-1] = 1.

    def __call__(self, f):
        """
        Apply the regularization operator to the input vector.

        Args:
            f (np.ndarray): Input vector.

        Returns:
            float or None: Sum of squares of Omega.dot(f), or None if shape mismatch.
        """
        if len(f) != self._Nx:
            return None
        else:
            return np.sum(self._Omega.dot(f)**2.0)
