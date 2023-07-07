import numpy as np
import pandas as pd
from pyscf import gto,scf,mcscf,mcpdft,fci,dft,tddft,mrpt

class SAMixTools():
    def get_mol(self):
        mol = gto.Mole()
        mol.atom = self._atom
        mol.symmetry = self.groupname
        mol.unit = "bohr"
        mol.charge = self.charge
        mol.basis = self.basis
        if self.hf_spin is None:
            mol.spin = np.sum(self.nelecas)%2
        else:
            mol.spin = self.hf_spin
        mol.build()
        return mol

    def make_sacaspdft(self):
        from sadmrgpdft import sadmrgpdft,symm
        mol = self.get_mol()
        mf = scf.ROHF(mol)
        syms = self.energies["sym"].tolist()
        mc = sadmrgpdft.make_caspdft(mf,self.ncas,self.nelecas,self.orbs,syms,do_scf=False)
        mc.ci = self.ci
        mc.mo_coeff = self.orbs
        return mc

    def get_mc(self,state):
        mol = self.get_mol()
        mc = mcscf.CASCI(mol,self.ncas,self.nelecas)
        mc.ci = self.ci[state]
        mc.mo_coeff = self.orbs
        return mc

    def get_sym_data(self):
        return self.energies["sym"]

    def get_solver_num(self,state):
        syms = self.get_sym_data()[0]
        gs_sym = syms[0]
        es_sym = syms[-1]
        nsyms = {}
        nsyms[gs_sym] = syms.count(gs_sym)
        if gs_sym != es_sym : nsyms[es_sym] = syms.count(es_sym)
        sym_state_n = state
        if sym_state_n < nsyms[gs_sym]: #gs (solver0)
            return 0
        elif sym_state_n >= nsyms[gs_sym]: #es (solver1)
            sym_state_n = sym_state_n - nsyms[gs_sym]
            return 1

    def get_fcisolver_spin(self,state):
        fcisolvers = self.make_sacaspdft().fcisolver.fcisolvers
        solver_num = self.get_solver_num(state)
        return fcisolvers[solver_num].spin

    def make_casdm1s(self,state):
        mc = self.make_sacaspdft()
        s = self.get_solver_num(state)
        return np.array(mc.fcisolver.fcisolvers[s].make_rdm1s(mc.ci[state],mc.ncas,mc.nelecas))

    def make_ss_nmos_nci_noccs(self,state):
        mc = ex.get_mc(state)
        mc.fcisolver.spin = self.get_fcisolver_spin(state)
        nmos,nci,noccs = mcscf.casci.cas_natorb(mc)
        return nmos,nci,noccs

    def make_ss_noccs(self,state):
        mc = self.get_mc(state)
        mc.fcisolver.spin = ex.get_fcisolver_spin(state)
        nmos,nci,noccs = mcscf.casci.cas_natorb(mc)
        nmos = nmos[:,mc.ncore:mc.ncore+mc.ncas]
        noccs = noccs[mc.ncore:mc.ncore+mc.ncas]
        return nmos,noccs

        # from dsk.mrdiagnostics.sa import get_dom_conf_and_weight
    def make_sa_nmos_nci_noccs(self):
        mc = self.make_sacaspdft()
        nmos, nci, noccs = mcscf.casci.cas_natorb(mc,verbose=0)
        return nmos,nci,noccs

    def get_sa_nmos_occs(self):
        nmos, nci, noccs = self.make_sa_nmos_nci_noccs()
        from dsk.mrdiagnostics.sa import get_dom_conf_and_weight
        gsoccs,gsw = get_dom_conf_and_weight(nci[self.gs_num],self.ncas,self.nelecas)
        esoccs,esw = get_dom_conf_and_weight(nci[self.es_num],self.ncas,self.nelecas)
        mc = self.make_sacaspdft()
        norbs = nmos[:,mc.ncore:mc.ncore+mc.ncas]
        return norbs,gsoccs,esoccs
    
    def write_molden(self,fn=None):
        norbs,gsoccs,esoccs = self.get_sa_nmos_occs()
        assert(norbs.shape[1] == len(gsoccs))
        energies = []
        for i in range(norbs.shape[1]):
            gsocc = int(gsoccs[i])
            esocc = int(esoccs[i])
            num = float(f"-{gsocc}.{esocc}")
            energies += [num]
        
        idx = np.argsort(energies)
        norbs = norbs[:,idx]
        energies = np.array(energies)[idx]
    
        from pyscf.tools import molden
        mol = self.get_mol()
        name = self.name
        if fn is None:
            fn = f"{name}.molden"
        molden.from_mo(mol,fn,norbs,ene=energies)

    def run_nevpt2(self):
        mc = self.make_sacaspdft()
        print("Running NEVPT2...")
        res = []
        twos_list = []
        for fcisolver in mc.fcisolver.fcisolvers:
            twos_list += [fcisolver.spin]*fcisolver.nroots
        for i in range(len(mc.ci)):
            nalpha, nbeta = mc.nelecas
            twos = twos_list[i]
            while nalpha - nbeta != twos:
                nalpha += 1
                nbeta -= 1
            nelecas = (nalpha,nbeta)
            mc2 = mcscf.CASCI(mc._scf,mc.ncas,nelecas)
            mc2.mo_coeff = mc.mo_coeff
            mc2.ci = mc.ci[i]
            mynevpt2 = mrpt.NEVPT(mc2)
            res += [mynevpt2.kernel()]
        res = np.array(res)
        self.energies["nevpt2"] = self.energies["mcscf"] + res
        self.data["energies"] = self.energies
        gs_nevpt2 = self.energies.loc[self.gs_num]["nevpt2"]
        es_nevpt2 = self.energies.loc[self.es_num]["nevpt2"]
        hartree_to_ev = 27.2114
        self.res["nevpt2"] = (es_nevpt2 - gs_nevpt2)*hartree_to_ev
        return self.res

