import time

from animate_plot import write_results2
from classeslib import *
from elasticity import *

time_start = time.time()

a = 1e-21  # creep material constant, [Pa]^n
n = 5  # creep material constant, [-]
temp = 353  # temperature, [K]
q = 35000  # creep activation energy, [cal/mol]
r = 1.987  # gas constant, [cal/(mol*K)]
mu = [11e9, 11e9, 11e9]  # Shear modulus, [Pa]
kb = [22e9, 22e9, 22e9]  # Bulk modulus, [Pa]
th = 1e3  # thickness of the domain, [m]
w = 1e2  # cavern's width, [m]
pr = 3e6  # difference between cavern's and lithostatic pressure, [Pa]
nt = 15  # number of time steps, [-]
dt = 31536000e-2  # time step, [s]

mesh = Mesh('./mesh/heterogen.msh')
sfns = Shapefns()
V = FunctionSpace(mesh, sfns, mu, kb)
k = V.stiff_matrix(th)
fo = V.load_vector(pr, w)
l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(mesh.coordinates(), dof=2)
d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))
k, fo = impose_dirichlet(k, fo, d_bnd)
u = np.linalg.solve(k, fo)
straing = V.gauss_strain(u)
strain = V.nodal_extrapolation(straing)
stressg = V.calc_stressg(straing)
stress = V.nodal_extrapolation(stressg)

disp_out = np.zeros((2 * mesh.nnodes(), nt))
stress_out = np.zeros((3 * mesh.nnodes(), nt))
strain_out = np.zeros((3 * mesh.nnodes(), nt))
forces_out = np.zeros((2 * mesh.nnodes(), nt))
svm_out = np.zeros((mesh.nnodes(), nt))
strain_crg = np.zeros((3, mesh.nele()))
et = [0]

disp_out[:, 0] = np.concatenate((u[::2].reshape(mesh.nnodes(), ), u[1::2].reshape(mesh.nnodes(), )), axis=0)
strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
svm_out[:, 0] = von_mises_stress(stress).transpose()

for i in range(nt - 1):
    f_cr, strain_crg = V.creep_load_vector2(dt, th, a, n, q, r, temp, stress, strain_crg)
    f = fo + f_cr
    f[d_bnd] = 0
    u = np.linalg.solve(k, f)
    straing = V.gauss_strain(u)
    strain = V.nodal_extrapolation(straing)
    stressg = V.calc_stressg(straing)
    stress = V.nodal_extrapolation(stressg)
    disp_out[:, i + 1] = np.concatenate((u[::2].reshape(mesh.nnodes(), ),
                                         u[1::2].reshape(mesh.nnodes(), )), axis=0)
    strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
    stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
    forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(mesh.nnodes(), ),
                                           f_cr[1::2].reshape(mesh.nnodes(), )), axis=0)
    svm_out[:, i + 1] = von_mises_stress(stress).transpose()

    # elapsed time
    et = np.append(et, et[-1] + dt)

disp_out = disp_out - disp_out[:, 0].reshape((2 * mesh.nnodes(), 1))
output = {
    'displacement': disp_out,
    'strain': strain_out,
    'stress': stress_out,
    'creep forces': forces_out,
    'Von Mises stress': svm_out,
    'elapsed time': et
}

elapsed = time.time() - time_start

write_results2(nt, mesh.coordinates(), mesh.cells(), output, 15, '.gif', exaggerate=False)
# plot_results(mesh.coordinates(), mesh.cells(), 'disp', 'strain', 'stress', u, strain, stress)

print("\nExplicit simulation is done in {} seconds.\n"
      "Total simulation time is {} days.\n"
      "\nMaximum displacement is {} m, creep displacement is {} m."
      .format(float("{0:.2f}".format(elapsed)),
              float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
              float("{0:.3f}".format(np.max(abs(output['displacement'])))),
              float("{0:.1e}".format(np.max(abs(output['displacement'][:, -1] - output['displacement'][:, 0]))))))
