import numpy as np
from ase import Atoms
from ase.io import read, write
from glob import glob

def rotation_matrix(n, t):
    n = np.array(n)
    n = n / np.linalg.norm(n)

    nx, ny, nz = n
    c = np.cos(t)
    s = np.sin(t)

    R = np.array([
        [c + nx*nx*(1-c),     nx*ny*(1-c) - nz*s, nx*nz*(1-c) + ny*s],
        [ny*nx*(1-c) + nz*s,  c + ny*ny*(1-c),    ny*nz*(1-c) - nx*s],
        [nz*nx*(1-c) - ny*s,  nz*ny*(1-c) + nx*s, c + nz*nz*(1-c)]
    ])

    return R

def shift_atoms(cell,atoms,target,static):
    
    min_distance = 999999
    shift        = np.array([0,0,0])
    for nx in [-1, 0, 1]:
        for ny in [-1, 0, 1]:
            for nz in [-1, 0, 1]:
                translate = nx*cell[0] + ny*cell[1] + nz*cell[2]

                ri = atoms[target]+translate

                rij = np.linalg.norm(atoms[static]-ri)
                
                if rij < min_distance: 
                    min_distance = rij 
                    shift = translate
    return np.array(shift)


# Molecule in fractional coordinates
file = 'CMP_MONOCLINIC_from_expTRICLINIC.vasp'
mol0 = read(file)

# Molecule in cartesian coordinates
write(
    'CONTCAR_mol0_cartesian.vasp',
    mol0,
    format="vasp",
    direct=False,
    vasp5=True
)

mol0 = read("CONTCAR_mol0_cartesian.vasp")

c0  = mol0.get_cell()
z0 = mol0.get_chemical_symbols()
r0 = mol0.get_positions()

'''
    c0 is Unit cell parameters
    z0 is Atoms 
    r0 is Atomic coordinates
    e0 is Types of atoms
    n0 is Number of types of atoms
    l0 is Labels
'''

e0 = list(set(z0)) 
e0 = sorted(e0)
n0 = [0 for i in range(len(e0))] 
l0 = []

for i in range(len(z0)):
    for j in range(len(e0)):
        if z0[i] == e0[j]: 
            n0[j] += 1
            l0.append(f'{z0[i]}{n0[j]}')

indx = {l0[i]:i for i in range(len(z0))}


# FIRST TWO O-H 
indx_point = [indx['H3'],indx['H5']]

indx_axis = {
    indx['H3']: [indx['C11'],indx['O9']],
    indx['H5']: [indx['C13'],indx['O11']],
}

thetas = [(np.pi/3)*t for t in range(6)]

for t1 in thetas:
    for t2 in thetas:

        indx_angle = {
            indx['H3']: t1,
            indx['H5']: t2,
        }

        new_r0 = r0.copy()

        for i in indx_point:

            ai = new_r0[indx_axis[i][0]].copy()
            aj = new_r0[indx_axis[i][1]].copy()
            pi = new_r0[i].copy()


            # Translate ai and aj to the pi unit cell
            T_ai = shift_atoms(c0,r0,indx_axis[i][0],i)
            T_aj = shift_atoms(c0,r0,indx_axis[i][1],i)

            ai += T_ai
            aj += T_aj

            mid = (ai+aj) * 0.5

            # Ensure the rotation axis pass through the origin
            ai -= mid 
            aj -= mid 
            pi -= mid 

            rot_axis = ai-aj

            R = rotation_matrix(rot_axis, indx_angle[i])
            new_r0[i] = R @ pi + mid  

        new_mol0 = mol0.copy()
        new_mol0.set_positions(new_r0)
        write(f'CMP_MONOCLINIC/CMP_MONOCLINIC_from_expTRICLINIC_{round(np.degrees(t1)):03d}_{round(np.degrees(t2)):03d}_000_000.vasp', new_mol0, format="vasp", vasp5=True, direct=False)


# SECOND TWO O-H 
indx_point = [indx['H6'],indx['H4']]

indx_axis = {
    indx['H6']: [indx['C14'],indx['O12']],
    indx['H4']: [indx['C12'],indx['O10']],
}

thetas = [(np.pi/3)*t for t in range(6)]

for t1 in thetas:
    for t2 in thetas:

        indx_angle = {
            indx['H6']: t1,
            indx['H4']: t2,
        }

        new_r0 = r0.copy()

        for i in indx_point:

            ai = new_r0[indx_axis[i][0]].copy()
            aj = new_r0[indx_axis[i][1]].copy()
            pi = new_r0[i].copy()


            # Translate ai and aj to the pi unit cell
            T_ai = shift_atoms(c0,r0,indx_axis[i][0],i)
            T_aj = shift_atoms(c0,r0,indx_axis[i][1],i)

            ai += T_ai
            aj += T_aj

            mid = (ai+aj) * 0.5

            # Ensure the rotation axis pass through the origin
            ai -= mid 
            aj -= mid 
            pi -= mid 

            rot_axis = ai-aj

            R = rotation_matrix(rot_axis, indx_angle[i])
            new_r0[i] = R @ pi + mid  

        new_mol0 = mol0.copy()
        new_mol0.set_positions(new_r0)
        write(f'CMP_MONOCLINIC/CMP_MONOCLINIC_from_expTRICLINIC_000_000_{round(np.degrees(t1)):03d}_{round(np.degrees(t2)):03d}.vasp', new_mol0, format="vasp", vasp5=True, direct=False)