# coords + 1e2*(displacement_x, [m]*iHat + displacement_y, [m]*jHat + 0*kHat)
import time
import xlsxwriter

from animate_plot import write_results
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
arrhenius = True
cfl = 0.8
g = 9.81  # gravity constant, [m/s^2]
rho = 2200  # average rock salt, [kg/m3]
depth = 700  # depth from the surface to the salt rock top interface, [m]
a = 8.1e-28  # creep material constant, [Pa^n]
n = 3.5  # creep material constant, [-]
temp = 298  # temperature, [K]
# q = 125000  # creep activation energy, [cal/mol]
q = 51600  # creep activation energy, [J/mol]
# r = 1.987  # gas constant, [cal/(mol*K)]
# r = 1.38e-23  # Boltzman constant, [m^2*kg*s^-2*K^-1]
r = 8.314  # gas constant, [J/mol/K]
# mu = np.array([7.5e9])  # Shear modulus, [Pa]
# kb = np.array([24.3e9])  # Bulk modulus, [Pa]
mu = 7.5e9  # Shear modulus, [Pa]
kb = 24.3e9  # Bulk modulus, [Pa]
lamda = kb - 2 / 3 * mu  # lame parameter
nu = (3 * kb - 2 * mu) / (2 * (3 * kb + mu))  # poisson's ratio
ym = 9 * kb * mu / (3 * kb + mu)  # young's moduli
th = 1  # thickness of the domain, [m]
nt = 30  # number of time steps, [-]
# dt = 31536000e-2  # time step, [s]
dt = 1e6  # time step, [s]
scale = 4e2
sign = 1

et = [0]
filename = 'new_cave2.msh'
mesh = Mesh('./mesh/' + filename, 4e2, 4e2)
sfns = Shapefns()
V = FunctionSpace(mesh, sfns, ym, nu)
k = V.stiff_matrix()
fo, sign = V.load_vector(rho * g, temp, g, depth, th, et[-1], 0, sign, pressure='min', boundary='cavern')
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

disp_out[:, 0] = np.concatenate((u[::2].reshape(mesh.nnodes(), ), u[1::2].reshape(mesh.nnodes(), )), axis=0)
strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
svm_out[:, 0] = von_mises_stress(stress).transpose()

g_cr = 3 / 2 * a * np.abs(np.power(von_mises_stress(stress), n - 2)) * von_mises_stress(
    stress) * deviatoric_stress(stress) * np.exp(- q / (r * temp))
# dt = cfl * 0.5 * np.max(np.abs(strain)) / np.max(np.abs(g_cr))

if NR == False:
    if nt > 1:
        for i in tqdm(range(nt - 1)):
            fo, sign = V.load_vector(rho * g, temp, g, depth, th, et[-1], i, sign, pressure='min', boundary='cavern')
            f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius)
            strain_cr = V.nodal_extrapolation(strain_crg)
            f = fo + f_cr
            f[d_bnd] = 0
            u = solve_disp(k, f)

            straing = V.gauss_strain(u)
            strain = V.nodal_extrapolation(straing)
            # stressg = V.gauss_stress(straing - strain_crg)
            # stress = V.nodal_extrapolation(stressg)

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
            # g_cr = 3 / 2 * a * np.abs(np.power(von_mises_stress(stress), n - 2)) * von_mises_stress(
            #     stress) * deviatoric_stress(stress) * np.exp(- q / (r * temp))
            # dt = cfl * 0.5 * np.max(np.abs(strain)) / np.max(np.abs(g_cr))
            # if dt < 3e6:
            #     dt *= 1.5

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

                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius)
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

# write_results(nt, mesh, output, filename.split(".")[0], '.xdmf', '.gif')
save_plot_A(nt, mesh, output, filename.split(".")[0], 700)

# nnodes = mesh.nnodes()
# workbook = xlsxwriter.Workbook('./output/' + filename.split(".")[0] + '/data.xlsx')
# worksheet = workbook.add_worksheet()
# row = 0
# col = 0
# order = sorted(output.keys())
# for key in order:
#     row += 1
#     worksheet.write(row, col, key)
#     for item in output[key]:
#         row += 1
#         i = 0
#         if type(item) == np.ndarray:
#             for value in item:
#                 worksheet.write(row, col + i, value)
#                 i += 1
#         else:
#             worksheet.write(row, col + i, item)
#             i += 1
#
#
# workbook.close()
