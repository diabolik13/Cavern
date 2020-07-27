# coords + 1e2*(displacement_x, [m]*iHat + displacement_y, [m]*jHat + 0*kHat)
import time
import sys

from animate_plot import write_results
from classeslib import *
from datetime import datetime
from elasticity import impose_dirichlet, solve_disp, deviatoric_stress, save_plot_A, write_xls
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
damage = False
cyclic = True

cfl = 0.8
g = 9.81  # gravity constant, [m/s^2]
rho = 2250  # average rock salt, [kg/m3]
depth = 600  # depth from the surface to the salt rock top interface, [m]
a = 8.1e-28  # creep material constant, [Pa^n]
bo = 1e-29  # material parameter
n = 3.5  # creep material constant, [-]
l = 4.5  # material parameter
kk = 4  # material parameter
temp = 298  # temperature, [K]
q = 51600  # creep activation energy, [J/mol]
r = 8.314  # gas constant, [J/mol/K]
ym = [44e9, 44e9, 44e9]
nu = [0.3, 0.3, 0.3]
th = 1  # thickness of the domain, [m]
nt = 5  # number of time steps, [-]
dt = 86400  # time step, [s]
scale = 4e2
sign = 1

et = [0]
filename = 'new_cave2.msh'
# filename = 'irreg2.msh'
mesh = Mesh('./mesh/' + filename, 4e2, 4e2)

# modeling impurity heterogeneity
imp2 = mesh.peak(100, -350, 150)
imp3 = mesh.onedpeak(25, -100)

imp_1 = np.zeros((mesh.nele(),))
imp_2 = np.zeros((mesh.nele(),))

for elt in range(mesh.nele()):
    nodes = mesh.cells(elt)
    imp_1[elt] = np.mean(imp2[nodes])
    imp_2[elt] = np.mean(imp3[nodes])

sfns = Shapefns()
V = FunctionSpace(mesh, sfns, ym, nu, imp_1, imp_2)
k = V.stiff_matrix()
fo, _ = V.load_vector(rho, temp, g, depth, pressure='min', boundary='cavern')
d_bnd = mesh.extract_bnd(lx=True, ly=False,
                         rx=True, ry=False,
                         bx=False, by=True,
                         tx=False, ty=False)
k, fo = impose_dirichlet(k, fo, d_bnd)
u = solve_disp(k, fo)

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

# damage evolution parameter
if damage == True:
    omega = np.zeros((mesh.nele(),))
    b = bo * np.exp(-q / r / temp)

# cyclic load stress
if cyclic == True:
    stressg1 = stressg
    stress1 = stress

if not NR:
    if nt > 1:
        for i in tqdm(range(nt - 1)):
            if cyclic:
                fo, sign = V.load_vector(rho, temp, g, depth, sign, i, 4, pressure='min', boundary='cavern')
            if damage:
                omega = b * (von_mises_stress(stressg).transpose()) ** n / (((1 - omega) * dt) ** l)
                s_I = 1 / 2 * (stressg[0] - stressg[1]) + 1 / 2 * np.sqrt(
                    (stressg[0] - stressg[1]) ** 2 + 4 * (stressg[2]) ** 2)
                svmg = von_mises_stress(stressg).transpose()
                sw_eq = 1 / 2 * (s_I + svmg)
                # sw_eq = svmg

            if damage:
                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius, omega)
            else:
                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius)

            strain_cr = V.nodal_extrapolation(strain_crg)
            f = fo + f_cr
            f[d_bnd] = 0
            u = solve_disp(k, f)

            straing = V.gauss_strain(u)
            strain = V.nodal_extrapolation(straing)

            if cyclic and i == 0:
                stressg2 = V.gauss_stress(straing - strain_crg)
                stress2 = V.nodal_extrapolation(stressg)

            if cyclic:
                if sign > 0:
                    stress = stress2
                    stressg = stressg2
                else:
                    stress = stress1
                    stressg = stressg1

            disp_out[:, i + 1] = np.concatenate((u[::2].reshape(mesh.nnodes(), ),
                                                 u[1::2].reshape(mesh.nnodes(), )), axis=0)
            strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
            stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
            forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(mesh.nnodes(), ),
                                                   f_cr[1::2].reshape(mesh.nnodes(), )), axis=0)
            svm_out[:, i + 1] = von_mises_stress(stress).transpose()
            strain_cr_out[:, i + 1] = np.concatenate((strain_cr[0], strain_cr[1], strain_cr[2]), axis=0)

            if damage and abs(np.max(omega)) >= 0.3:
                disp_out[:, i + 1] = float('nan')
                strain_out[:, i + 1] = float('nan')
                stress_out[:, i + 1] = float('nan')
                forces_out[:, i + 1] = float('nan')
                svm_out[:, i + 1] = float('nan')
                strain_cr_out[:, i + 1] = float('nan')

            # elapsed time
            et = np.append(et, et[-1] + dt)

            # g_cr = 3 / 2 * a * np.abs(np.power(von_mises_stress(stress), n - 2)) * von_mises_stress(
            #     stress) * deviatoric_stress(stress) * np.exp(- q / (r * temp))
            # dt = cfl * 0.5 * np.max(np.abs(strain)) / np.max(np.abs(g_cr))
            # if dt < 3e6:
            #     dt *= 1.5
            # if np.max(omega) < 1 and np.max(omega) > 0.1:
            #     dt = 0.8*(1e6 - 1e6 * np.max(omega))
            # print('omega=' + str(np.max(omega)))

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

# save 2D results
write_results(nt, mesh, output, filename.split(".")[0], '.gif')
# save parameters in point A evolution in time
# save_plot_A(nt, mesh, output, filename.split(".")[0], 700)
# write results to xls file
# write_xls(filename, output)
