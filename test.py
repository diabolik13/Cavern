import time

from animate_plot import write_results2, write_results_xdmf2
from classeslib import *
from elasticity import *
from tqdm import tqdm


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("log.txt")
time_start = time.time()

# c1 = 0.5
# c2 = - np.pi / 2
c1 = 1
c2 = - np.pi / 2

a = 1e-25  # creep material constant, [Pa]^n
n = 5  # creep material constant, [-]
temp = 343  # temperature, [K]
q = 35000  # creep activation energy, [cal/mol]
r = 1.987  # gas constant, [cal/(mol*K)]
mu = np.array([12e9, 12e9, 12e9])  # Shear modulus, [Pa]
kb = np.array([21e9, 21e9, 21e9])  # Bulk modulus, [Pa]
lamda = kb - 2 / 3 * mu
th = 1  # thickness of the domain, [m]
# w = 1  # cavern's width, [m]
pr = -13e6  # difference between cavern's and lithostatic pressure, [Pa]
nt = 50  # number of time steps, [-]
dt = 31536000e-2  # time step, [s]
d = np.array([[lamda[0] + 2 * mu[0], lamda[0], 0],
              [lamda[0], lamda[0] + 2 * mu[0], 0],
              [0, 0, mu[0]]])

filename = 'testcase.msh'
mesh = Mesh('./mesh/' + filename, 1e3, 1e3)
# anl_solution = anl(c1, c2)
# u_anl = anl_solution.evaluate_displacement(mesh)
# strain = anl_solution.evaluate_strain(mesh)
# f_anl = anl_solution.evaluate_forces(mesh, lamda[0], mu[0])

sfns = Shapefns()
V = FunctionSpace(mesh, sfns, mu, kb)
k = V.stiff_matrix(th)
# ko = assemble_stiffness_matrix(2, mesh.coordinates(), mesh.cells(), d, th)
fo = V.load_vector2(pr, th)
d_bnd = mesh.extract_bnd(lx=True, ly=False,
                         rx=False, ry=True,
                         bx=False, by=True,
                         tx=False, ty=True)
k, fo = impose_dirichlet(k, fo, d_bnd)
u = solve_disp(k, fo)
plot_parameter(mesh, fo[::2], filename.split(".")[0])
# plot_parameter(mesh, fo[1::2], filename.split(".")[0])

straing = V.gauss_strain(u)
strain = V.nodal_extrapolation(straing)
stressg = V.gauss_stress(straing)
stress = V.nodal_extrapolation(stressg)

disp_out = np.zeros((2 * mesh.nnodes(), nt))
stress_out = np.zeros((3 * mesh.nnodes(), nt))
strain_out = np.zeros((3 * mesh.nnodes(), nt))
forces_out = np.zeros((2 * mesh.nnodes(), nt))
svm_out = np.zeros((mesh.nnodes(), nt))
strain_crg = np.zeros((3, mesh.nele()))
strain_cr_out = np.zeros((3 * mesh.nnodes(), nt))
et = [0]

disp_out[:, 0] = np.concatenate((u[::2].reshape(mesh.nnodes(), ), u[1::2].reshape(mesh.nnodes(), )), axis=0)
strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
svm_out[:, 0] = von_mises_stress(stress).transpose()

if nt > 1:
    for i in tqdm(range(nt - 1)):
        f_cr, strain_crg = V.creep_load_vector2(dt, th, a, n, q, r, temp, stressg, strain_crg)
        strain_cr = V.nodal_extrapolation(strain_crg)
        f = fo + f_cr
        f[d_bnd] = 0
        u = solve_disp(k, f)
        straing = V.gauss_strain(u)
        strain = V.nodal_extrapolation(straing)
        stressg = V.gauss_stress(straing - strain_crg)
        stress = V.nodal_extrapolation(stressg)
        disp_out[:, i + 1] = np.concatenate((u[::2].reshape(mesh.nnodes(), ),
                                             u[1::2].reshape(mesh.nnodes(), )), axis=0)
        strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
        stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
        forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(mesh.nnodes(), ),
                                               f_cr[1::2].reshape(mesh.nnodes(), )), axis=0)
        svm_out[:, i + 1] = von_mises_stress(stress).transpose()
        strain_cr_out[:, i + 1] = np.concatenate((strain_cr[0], strain_cr[1], strain_cr[2]), axis=0)

        # elapsed time
        et = np.append(et, et[-1] + dt)

# displacement due to creep only
# disp_out = disp_out - disp_out[:, 0].reshape((2 * mesh.nnodes(), 1))

output = {
    'displacement': disp_out,
    'strain': strain_out,
    'stress': stress_out,
    'creep forces': forces_out,
    'Von Mises stress': svm_out,
    'elapsed time': et
}

elapsed = time.time() - time_start

print("\nExplicit simulation is done in {} seconds.\n"
      "Total simulation time is {} days.\n"
      "\nMaximum elastic displacement is {} m, creep displacement is {} m."
      .format(float("{0:.2f}".format(elapsed)),
              float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
              float("{0:.3f}".format(np.max(abs(output['displacement'][:, 0])))),
              float("{0:.1e}".format(np.max(abs(output['displacement'][:, -1] - output['displacement'][:, 0]))))))

write_results_xdmf2(nt, mesh, output, filename.split(".")[0])
# write_results2(nt, mesh.coordinates(), mesh.cells(), output, 10, '.gif', exaggerate=False)
