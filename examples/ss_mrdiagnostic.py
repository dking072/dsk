from pyscf import gto,scf,mcscf

mol = gto.Mole(atom=["H 0 0 0","H 0 0 1"])
mol.build()
mf = scf.RHF(mol)
mf.kernel()
mc = mcscf.CASCI(mf,2,2)
mc.kernel()

from dsk.mrdiagnostics import ss as mrd

#Entropies -- sum of entropies / 1.4 is called the "Z diagnostic" 10.1002/jcc.25869
#M diagnostic -- deviation of natural orbitals from dominant configuration 10.1021/ct800077r

ndist = mrd.calc_n_dist(mc) #orbital occupancy probability distribution 0/u/d/2
entropies = mrd.calc_entropies(mc) #orbital entropies (all orbitals, non-active are 0)
entropies[mc.ncore:mc.ncore+mc.ncas] #only the active orbitals

m = mrd.calc_mdiagnostic(mc) #M diagnostic (slow because of natorb transform)

#Alternatively, compute the noccs first for other use:
nmos,nci,noccs = mcscf.casci.cas_natorb(mc)
m = mrd.calc_mdiagnostic_withnci(mc,nci,noccs)