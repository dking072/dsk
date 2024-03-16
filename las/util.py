import numpy as np
from pyscf import gto, scf, lib, mcscf
import time
import pandas as pd
import math
import seaborn as sns

def las_charges(las):
    las_charges = [[fcisolver.charge for fcisolver in las.fciboxes[i].fcisolvers] for i in range(len(las.fciboxes))]
    las_charges = np.array(las_charges).T
    return las_charges