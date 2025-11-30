from .noregularization import NoRegularization
from .tikhonovorder0 import TikhonovOrder0
from .tikhonovorder1 import TikhonovOrder1
from .tikhonovorder2 import TikhonovOrder2
from .maxentropy0 import MaxEntropy0
from .maxentropy1 import MaxEntropy1
from .maxentropy2 import MaxEntropy2

__all__ = [
    "NoRegularization",
    "TikhonovOrder0",
    "TikhonovOrder1",
    "TikhonovOrder2",
    "MaxEntropy0",
    "MaxEntropy1",
    "MaxEntropy2",
]
