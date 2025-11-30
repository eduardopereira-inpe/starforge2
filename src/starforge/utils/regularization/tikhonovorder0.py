import numpy as np
from itertools import islice, count

class TikhonovOrder0(object):
    """
    Tikhonov regularization of order 0 (identity matrix).
    """

    def __init__(self):
        self.name = 'Tikhonov Order 0'
        self.type = 'tikhonovorder0'

    def __call__(self, f):
        return np.sum(f**2)
   