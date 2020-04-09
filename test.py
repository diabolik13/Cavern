import numpy as np
from classeslib import *
from elasticity import *

# mesh_filename = 'new_cave.msh'
# input_param = load_input(mesh_filename)
# k1 = input_param['stiffness matrix']
# f1 = input_param['external forces']

mu = 11e9  # Shear modulus, [Pa]
kb = 22e9  # Bulk modulus, [Pa]
th = 1e3  # thickness of the domain, [m]
w = 1e2  # cavern's width, [m]
pr = 3e6  # difference between cavern's and lithostatic pressure, [Pa]

mesh = Mesh('./mesh/new_cave.msh')
d = el_tenzor(mu, kb)
sfns = Shapefns()
V = FunctionSpace(mesh, sfns)
k = V.stiff_matrix(mu, kb, th)
f = V.load_vector(mesh, pr, w)
l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(mesh.coordinates(), dof=2)
d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))
k, f = impose_dirichlet(k, f, d_bnd)

u = np.linalg.solve(k, f)
straing, stressg = gauss_stress_strain(mesh.coordinates(), mesh.cells(), u, d)
strain, stress = nodal_stress_strain(mesh.coordinates(), mesh.cells(), straing, stressg)

plot_results(mesh.coordinates(), mesh.cells(), 'disp', 'strain', 'stress', u, strain, stress)
print(str(mesh.Ndofs()))
