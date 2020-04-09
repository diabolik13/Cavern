from elasticity import *
from animate_plot import *

c1 = 0
c2 = 0
folder = './mesh/'
mesh_filename = folder + 'rect.msh'

input_param = load_input(mesh_filename, c1, c2)
# f = input_param['external forces']
# k = input_param['stiffness matrix']
p = input_param['points']
t = input_param['elements']
d = input_param['elasticity tensor']
lamda = input_param['Lame parameter']
mu = input_param['shear moduli']
nnodes = p.shape[1]
x = p[0, :]
y = p[1, :]

f = assemble_vector(p, t, 0, 0, 1000, simple=True)
l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(p, 2)
d_bnd = np.concatenate((b_bnd, l_bnd))
k = assemble_stiffness_matrix(2, p, t, d, 1000)
k, f = impose_dirichlet(k, f, d_bnd)

u = np.linalg.solve(k, f).reshape((2 * nnodes, 1))
straing, stressg = gauss_stress_strain(p, t, u, d)
strain, stress = nodal_stress_strain(p, t, straing, stressg)

plot_results2(p, t, 'disp', 'strain', 'stress', u, strain, stress)
