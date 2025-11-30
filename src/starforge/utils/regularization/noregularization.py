import numpy as np

class NoRegularization(object):
    """
    No regularization (identity operator).

    This class implements a trivial regularization operator that returns the L2 norm
    of the input vector after applying a zero matrix (always zero).

    Attributes:
        Nx (int): Size of the input vector.
        Omega (np.ndarray): Zero matrix of shape (Nx, Nx).
    """

    def __init__(self, Nx):
        """
        Args:
            Nx (int): Size of the input vector.
        """
        self.Nx = Nx
        self.Omega = np.zeros((self.Nx, self.Nx))

    def __call__(self, f):
        """
        Apply the regularization operator to the input vector.

        Args:
            f (np.ndarray): Input vector.

        Returns:
            float or None: L2 norm of Omega.dot(f), or None if shape mismatch.
        """
        if len(f) != self.Nx:
            return None
        else:
            return np.linalg.norm(self.Omega.dot(f), 2)
