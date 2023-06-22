import os
import pickle
import numpy as np
import glob
import pandas as pd
import time
from pyscf import gto, scf, mcscf, lib, symm, ao2mo, lo, fci
from pyscf.mcscf.addons import state_average_mix,state_average_mix_

def get_sym_data(mc):
    syms = []
    for fcisolver in mc.fcisolver.fcisolvers:
        spacesym = fcisolver.wfnsym
        spinsym = fcisolver.spin + 1
        syms += [f"{spinsym}{spacesym}"]*fcisolver.nroots
    return (syms,mc.mol.symmetry)

def sa_calc_n_dist(mc):
    """
    Results returned as 0,u,d,2!
    """
    results = {}
    for state in range(len(mc.ci)):
        #Only coding for two fcisolvers right now:
        assert(len(mc.fcisolver.fcisolvers) <= 2)
        if state <= mc.fcisolver.fcisolvers[0].nroots - 1:
            fcisolver = mc.fcisolver.fcisolvers[0]
        else:
            fcisolver = mc.fcisolver.fcisolvers[1]

        #Get 1 and 2 rdm for the state, calculate occ dist for all orbitals:
        casdm1s,casdm2s = fcisolver.make_rdm12s(mc.ci[state],mc.ncas,mc.nelecas)
        casdm2aa,casdm2ab,casdm2bb = casdm2s
        orb_dists = np.zeros([mc.ncas,4])
        for i in range(mc.ncas):
            ga, gb = casdm1s
            rho00 = 1 - ga[i,i] - gb[i,i] + casdm2ab[i,i,i,i]
            rhouu = ga[i,i] - casdm2ab[i,i,i,i]
            rhodd = gb[i,i] - casdm2ab[i,i,i,i]
            rho22 = casdm2ab[i,i,i,i]
            diag = [rho00,rhouu,rhodd,rho22]
            assert(np.allclose(np.sum(diag),1))
            diag = np.array(diag) + 1e-36 #Salt to avoid div0 errors in entropy calc
            assert((np.array(diag) > 0).all())
            orb_dists[i,:] = diag
        spin = fcisolver.spin + 1
        wfnsym = fcisolver.wfnsym
        sym = f"{spin}{wfnsym}-{state}"
        df = pd.DataFrame(orb_dists,columns=[0,"u","d",2])
        results[sym] = pd.DataFrame(orb_dists,columns=[0,"u","d",2])
    return results

def sa_calc_entropies(mc):
    sa_dists = sa_calc_n_dist(mc)
    sa_dists = {int(k[-1]):v for k,v in sa_dists.items()}
    assert(len(sa_dists.keys()) == len(mc.ci))
    
    #Calculate the entropies of each AS orbital in each state
    df2 = pd.DataFrame()
    for k,df in sa_dists.items():
        entropies = -df[0]*np.log(df[0])
        entropies += -df["u"]*np.log(df["u"])
        entropies += -df["d"]*np.log(df["d"])
        entropies += -df[2]*np.log(df[2])
        entropies = [0]*mc.ncore + list(entropies) + [0]*(mc.mol.nao-mc.ncore-mc.ncas)
        entropies = pd.Series(entropies)
        df2 = df2.append(entropies,ignore_index=True)
    return df2

def get_dom_conf_and_weight(ci,ncas,nelecas):
    from pyscf.fci import cistring
    
    #https://github.com/pyscf/pyscf/issues/145
    a_addr,b_addr = np.unravel_index(np.argmax(np.abs(ci), axis=None), ci.shape)
    na,nb = nelecas
    astr = cistring.addrs2str(ncas,na,[a_addr])
    bstr = cistring.addrs2str(ncas,nb,[b_addr])
    a_occslst = cistring._strs2occslst(astr,ncas)
    b_occslst = cistring._strs2occslst(bstr,ncas)
    occs = np.zeros(ncas)
    occs[a_occslst] += 1
    occs[b_occslst] += 1
    weight = np.abs(ci[a_addr,b_addr])
    return occs, weight

def calc_mdiagnostic(mc,ci,spin,verbose=False):
    mc2 = mcscf.CASCI(mc,mc.ncas,mc.nelecas)
    mc2.ci = ci
    mc2.fcisolver.spin = spin
    nmos,nci,noccs = mcscf.casci.cas_natorb(mc2)
    act_start = (mc2.mol.nelectron - np.sum(mc2.nelecas))//2
    act_stop = act_start + mc2.ncas
    noccs = noccs[act_start:act_stop]
    
    dom_occs,weight = get_dom_conf_and_weight(nci,mc2.ncas,mc2.nelecas)

    donos = noccs[np.where(dom_occs == 2)]
    somos = noccs[np.where(dom_occs == 1)]
    unos = noccs[np.where(dom_occs == 0)]

    mcdono = np.min(donos)
    mcsomos = np.sum(np.abs(somos - 1))
    mcuno = np.max(unos)

    M = 0.5*(2 - mcdono + mcsomos + mcuno)

    if verbose:
        print("noccs:",np.round(noccs,2))
        print("dom_occs:",dom_occs,weight)
        print("M:",M)
    return M

def sa_calc_mdiagnostic(mc,verbose=False):
    mdiagnostics = []
    syms = get_sym_data(mc)[0]
    for state in range(len(mc.ci)):
        sym = syms[state]
        twosplus1 = int(sym[0])
        twos = twosplus1 - 1
        if verbose : print("State",state,":")
        m = calc_mdiagnostic(mc,mc.ci[state],twos,verbose=verbose)
        mdiagnostics += [m]
    return mdiagnostics
