import numpy as np

def make_nos(c,s,dao):
    """
    c --> mo coefficients (any orthogonal basis works)
    s --> ovlp matrix
    dm --> 1rdm
    returns noccs and norbs in ao basis
    """
    dmo = np.linalg.multi_dot([c.T,s,dao,s,c])
    noccs,norbs = np.linalg.eigh(dmo)
    norbs = np.matmul(c,norbs)
    return noccs, norbs
