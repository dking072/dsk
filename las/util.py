import numpy as np
from pyscf import gto, scf, lib, mcscf
import time
import pandas as pd
import math
import seaborn as sns
from dsk.pickle import load_pkl, dump_pkl
from dsk.las import sign_control, hcircle, util, bandh

def las_charges(las):
    las_charges = [[fcisolver.charge for fcisolver in las.fciboxes[i].fcisolvers] for i in range(len(las.fciboxes))]
    las_charges = np.array(las_charges).T
    return las_charges

class LASdata:
    def __init__(self,pkl_fn):
        data = load_pkl(pkl_fn)
        energies = data["energies"]
        civecs = data["civecs"]
        charges = data["charges"]
        self.data = data
        self.hdct = bandh.make_hdct(civecs,energies,charges)

    def get_homo(self):
        e,k = bandh.calc_homo(self.hdct).values()
        return e,k

    def get_lumo(self):
        e,k = bandh.calc_lumo(self.hdct).values()
        return e,k

class LASHdata(LASdata):

    def hf_homo(self):
        mf_coeff = self.data["mf_coeff"]
        mo_occ = self.data["mf_occ"]
        energies = self.data["mf_ene"]
        homo_idx = np.where(mo_occ == 2)[0]
        mos = mf_coeff[:,homo_idx]
        energies = energies[homo_idx]
        k = bandh.calc_disp(mos) * 2
        energies,k = bandh.copy_bz(energies,k)
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
        return energies,k

    def hf_lumo(self):
        mf_coeff = self.data["mf_coeff"]
        mo_occ = self.data["mf_occ"]
        energies = self.data["mf_ene"]
        lumo_idx = np.where(mo_occ == 0)[0]
        mos = mf_coeff[:,lumo_idx]
        energies = energies[lumo_idx]
        k = bandh.calc_disp(mos)
        k = k[::-1]*2 - 0.5 #For antibonding
        energies,k = bandh.copy_bz(energies,k)
        hartree_to_ev = 27.2114
        energies *= hartree_to_ev
        return energies,k