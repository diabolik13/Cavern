import time

from elasticity import *
from animate_plot import *

time_start = time.time()

# Define input parameters and load mesh
mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
rho = 2160  # rock density, [kg/m3]
K = 22e9  # Bulk modulus, [Pa]
mu = 11e9  # Shear modulus, [Pa]
P = 3e6  # cavern's pressure, [Pa]
dof = 2  # degrees of freedom, [-]
Nt = 15  # number of time steps, [-]
A = 1e-42  # creep material constant, [Pa]^n
n = 5  # creep material constant, [-]
th = 1e3  # thickness of the model in z, [m]
w = 1e2  # cavern width in z, [m]
dt = 31536000e-6  # time step, [s]
c = 0  # wave number, number of cycles, [-]
cfl = 0.5  # CFL

m, p, t = load_mesh(mesh_filename)
lamda, E, nu, D = lame(K, mu, plane_stress=True)
L_bnd, R_bnd, B_bnd, T_bnd = extract_bnd(p, dof)
D_bnd = np.concatenate((B_bnd, T_bnd, L_bnd, R_bnd))
Px, Py, nind_c = cavern_boundaries(m, p, P, w)
k = assemble_stiffness_matrix(dof, p, t, D, th)
f = assemble_vector(p, t, nind_c, Px, Py)
k, f = impose_dirichlet(k, f, D_bnd)
# check_matrix(k)

input = {
    'time step size': dt,
    'number of timesteps': Nt,
    'thickness': th,
    'points': p,
    'elements': t,
    'material constant': A,
    'material exponent': n,
    'elasticity tensor': D,
    'shear moduli': mu,
    'Lame parameter': lamda,
    'external forces': f,
    'stiffness matrix': k,
    'Dirichlet boundaries': D_bnd,
    'CFL': cfl,
    'wave number': c
}

output = calculate_creep(input, m, P, w)
output_NR = calculate_creep_NR(input, m, P, w)
diff = np.max(abs(output['displacement']-output_NR['displacement']))
elapsed = time.time() - time_start
# print("Simulation is done in {} seconds. Total simulation is {} days. "
#       "Maximum displacement is {} m, creep displacement is {} m."
#       .format(float("{0:.2f}".format(elapsed)),
#               float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
#               float("{0:.3f}".format(np.max(abs(output['displacement'])))),
#               float("{0:.1e}".format(np.max(abs(output['displacement'][:][-1] - output['displacement'][:][0]))))))

# write_results_gif(Nt, p, t, output, 15, '.gif', exaggerate=False)
# write_results_xdmf(Nt, m, p, output)
print("Done writing results to output files.")
