import numpy as np
import os
from itertools import islice, count

class MaxEntropy2(object):
    """
    Maximum entropy regularization of order 2 (second difference).

    Attributes:
        name (str): Name of the regularization.
        type (str): Type identifier.
        _Nx (int): Size of the input vector.
        _Omega (np.ndarray): Second difference matrix.
        _Smax (float): Maximum entropy normalization.
        _chi (float): Small positive constant for numerical stability.
    """

    def __init__(self, Nx):
        """
        Args:
            Nx (int): Size of the input vector.
        """
        self.name = 'Max Entropy Order 2'
        self.type = 'maxentropy2'

        self._Nx = Nx
        self._Omega = np.zeros((self._Nx, self._Nx))
        self._Smax = np.log(self._Nx-2)
        self._chi = 0.1

        for i in islice(count(), 1, self._Nx-1):
            self._Omega[i][i] = -2.
            if i + 1 < Nx:
                self._Omega[i][i+1] = 1.
            if i - 1 >= 0:
                self._Omega[i][i-1] = 1.

        print(self._Omega)

    def __call__(self, f):
        """
        Apply the regularization operator to the input vector.

        Args:
            f (np.ndarray): Input vector.

        Returns:
            float or None: Normalized entropy value, or None if shape mismatch.
        """
        if len(f) != self._Nx:
            return None
        else:
            fmin = np.min(f)
            fmax = np.max(f)
            p = self._Omega.dot(f) + 2*(fmax-fmin) + self._chi
            if np.min(p) < 0:
                print(p)
                print('alert')
                os.sys.exit('1')
            psum = np.sum(p[1:self._Nx-1])
            s = p/psum
            return 1.0 - np.sum(s*np.log(s))/self._Smax
