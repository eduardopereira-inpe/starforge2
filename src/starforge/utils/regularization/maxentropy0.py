import numpy as np

class MaxEntropy0(object):
    """
    Maximum entropy regularization of order 0.

    Attributes:
        name (str): Name of the regularization.
        type (str): Type identifier.
        _Nx (int): Size of the input vector.
        _Smax (float): Maximum entropy normalization.
        _chi (float): Small positive constant for numerical stability.
    """

    def __init__(self, Nx):
        """
        Args:
            Nx (int): Size of the input vector.
        """
        self.name = 'Max Entropy Order 0'
        self.type = 'maxentropy0'

        self._Nx = Nx
        self._Smax = np.log(self._Nx)
        self._chi = 0.1

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
            p = f - fmin + self._chi
            psum = np.sum(p)
            s = p/psum
            return 1.0 - np.sum(s*np.log(s))/self._Smax
