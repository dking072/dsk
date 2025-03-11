def mol_to_xyz(mol,xyz_fn):
    natom = mol.natm
    xyz = mol.atom_coords(unit="ang")
    els = mol.elements
    lines = [f"{natom}\n","\n"]
    for el, xyz in zip(els,xyz):
        lines += [f"{el} {xyz[0]} {xyz[1]} {xyz[2]}\n"]
    lines += ["\n"]
    with open(xyz_fn,"w+") as f:
        for line in lines:
            f.write(line)