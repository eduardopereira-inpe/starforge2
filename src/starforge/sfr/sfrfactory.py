# sfr_factory.py
from starforge.structures import HaloBaryonAccretion
from .sfr import SFR

class SFRFactory:
    def __init__(self, massfunction):
        self.massfunction = massfunction

    def create_halo_accretion(self, zmax=20.0):
        return HaloBaryonAccretion(
            massfunction=self.massfunction,
            zmax=zmax
        )

    def create_sfr(
        self,
        tau=2.29,
        eimf=1.35,
        nsch=1,
        imfType="S",
        zmax=20.0,
        halo_acc=None
    ):
        if halo_acc is None:
            halo_acc = self.create_halo_accretion(zmax=zmax)

        return SFR(
            halo_acc=halo_acc,
            tau=tau,
            eimf=eimf,
            nsch=nsch,
            imfType=imfType,
            zmax=zmax
        )
