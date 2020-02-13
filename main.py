import time

from elasticity import *
from animate_plot import *

time_start = time.time()

# Define input parameters and load mesh
mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
rho = 2160  # rock density, [kg/m3]
K = 22e9  # Bulk modulus, [Pa]
mu = 11e9  # Shear modulus, [Pa]
P = 5e6  # cavern's pressure, [Pa]
dof = 2  # degrees of freedom, [-]
Nt = 3  # number of time steps, [-]
A = 1e-42  # creep material constant, [Pa]^n
n = 5  # creep material constant, [-]
th = 1e3  # thickness of the model, [m]
w = 1e2  # cavern width (used for cavern boundary forces calculation), [m]
dt = 31536000e-2  # time step, [s]
cfl = 1e3  # CFL

lamda, E, nu, D = lame(K, mu)
m, p, t = load_mesh(mesh_filename)
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
    'CFL': cfl
}

output = calculate_creep(input)
elapsed = time.time() - time_start
print("Simulation is done in {} seconds. Total simulation is {} seconds. "
      "Maximum displacement is {} m.".format(elapsed, output['elapsed time'][-1], np.max(abs(output['displacement']))))

write_results_gif(Nt, p, t, output)
write_results_xdmf(Nt, m, p, output)
