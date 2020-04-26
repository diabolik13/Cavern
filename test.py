# coords + 90*(displacement_x, [m]*iHat + displacement_y, [m]*jHat + 0*kHat)
import time

from animate_plot import write_results2, write_results_xdmf2
from classeslib import *
from datetime import datetime
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

NR = False
g = 9.81  # gravity constant, [m/s^2]
rho = 2750  # average rock salt, [kg/m3]
depth = 300  # depth from the surface to the salt rock top interface, [m]
a = 1e-25  # creep material constant, [Pa^n]
n = 5  # creep material constant, [-]
temp = 343  # temperature, [K]
q = 35000  # creep activation energy, [cal/mol]
r = 1.987  # gas constant, [cal/(mol*K)]
mu = np.array([12e9, 12e9, 12e9])  # Shear modulus, [Pa]
kb = np.array([21e9, 21e9, 21e9])  # Bulk modulus, [Pa]
lamda = kb - 2 / 3 * mu
th = 1  # thickness of the domain, [m]
nt = 50  # number of time steps, [-]
dt = 31536000e-2  # time step, [s]

filename = 'new_cave2.msh'
mesh = Mesh('./mesh/' + filename, 1e3, 1e3)
sfns = Shapefns()
V = FunctionSpace(mesh, sfns, mu, kb)
k = V.stiff_matrix()
fo = V.load_vector(rho * g, th, pressure='min', boundary='cavern')
d_bnd = mesh.extract_bnd(lx=True, ly=False,
                         rx=True, ry=False,
                         bx=False, by=True,
                         tx=False, ty=False)
k, fo = impose_dirichlet(k, fo, d_bnd)
u = solve_disp(k, fo)
# plot_parameter(mesh, fo[::2], filename.split(".")[0])
# plot_parameter(mesh, fo[1::2], filename.split(".")[0])

straing = V.gauss_strain(u)
strain = V.nodal_extrapolation(straing)
stressg = V.gauss_stress(straing)
stress = V.nodal_extrapolation(stressg)

dstress = np.zeros((3, mesh.nnodes()))
dstrain_cr = np.zeros((3, mesh.nnodes()))
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

if NR == False:
    if nt > 1:
        for i in tqdm(range(nt - 1)):
            f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg)
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

if NR == True:
    J = k
    if nt > 1:
        for i in tqdm(range(nt - 1)):
            # print('\nTime step {}, dt = {} s:'.format(i + 1, dt))
            converged = 0
            iter = 0
            max_iter = 5
            conv = 1e-4

            while converged == 0:
                # mesh.update_mesh(u)
                # FunctionSpace(mesh, sfns, mu, kb)

                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg)
                strain_cr = V.nodal_extrapolation(strain_crg)
                f = fo + f_cr
                f[d_bnd] = 0
                residual = np.dot(k, u) - f
                delta_u = - np.linalg.solve(J, residual)
                u = u + delta_u

                straing = V.gauss_strain(u)
                strain = V.nodal_extrapolation(straing)
                stressg = V.gauss_stress(straing - strain_crg)
                stress = V.nodal_extrapolation(stressg)
                et = np.append(et, et[-1] + dt)

                residual = np.dot(k, u) - f
                res = np.linalg.norm(residual)
                iter += 1

                # print("\nIteration {}, norm(residual) = {}.".format(iter, res))
                if iter == max_iter and res >= conv:
                    print("\nMaximum iterations reached.")
                if res < conv or iter >= max_iter:
                    converged = 1

                if converged == 1:
                    disp_out[:, i + 1] = np.concatenate((u[::2].reshape(mesh.nnodes(), ),
                                                         u[1::2].reshape(mesh.nnodes(), )), axis=0)
                    strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
                    stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
                    forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(mesh.nnodes(), ),
                                                           f_cr[1::2].reshape(mesh.nnodes(), )), axis=0)
                    svm_out[:, i + 1] = von_mises_stress(stress).transpose()
                    strain_cr_out[:, i + 1] = np.concatenate((strain_cr[0], strain_cr[1], strain_cr[2]), axis=0)

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
print("\n" + str(datetime.now()))

if NR == False:
    print("\nExplicit simulation is done in {} seconds.\n".format(float("{0:.2f}".format(elapsed))))
if NR == True:
    print("\nImplicit simulation is done in {} seconds.\n".format(float("{0:.2f}".format(elapsed))))

print("Total simulation time is {} days.\n Maximum elastic displacement is {} m, creep displacement is {} m.\n".format(
    float("{0:.2f}".format((output['elapsed time'][-1] / 86400))),
    float("{0:.3f}".format(np.max(abs(output['displacement'][:, 0])))),
    float("{0:.1e}".format(np.max(abs(output['displacement'][:, -1] - output['displacement'][:, 0]))))))

write_results_xdmf2(nt, mesh, output, filename.split(".")[0])
# write_results2(nt, mesh.coordinates(), mesh.cells(), output, 10, '.gif', exaggerate=False)
