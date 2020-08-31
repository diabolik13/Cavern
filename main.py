# for paraview post processing: coords + (displacement_x, [m]*iHat + displacement_y, [m]*jHat + 0*kHat)
import time

from animate_plot import write_results
from classeslib import *
from datetime import datetime
from elasticity import impose_dirichlet, solve_disp, deviatoric_stress, save_plot_A, write_xls
from tqdm import tqdm

sys.stdout = Logger("log.txt")
time_start = time.time()

NR = False
arrhenius = True
damage = False
cyclic = False
impurities = False
principal_stress = False
pressure = 'min'  # switch between minumum 'min' and maximum 'max' pressure of the cavern
boundary = 'cavern'  # whenere to apply Neumann bc ('cavern', 'right', 'top')

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
th = 1  # thickness of the domain, [m]
nt = 25  # number of time steps, [-]
dt = 1e6  # time step size, [s]
scale = 3e2  # scale the domain from -1...1 to -X...X, where X - is the scale
sign = 1  # used for cyclic load
imp_1 = None
imp_2 = None

if impurities == True:
    ym = [7e8, 24e9, 44e9]  # Young's modulus, [Pa] for 3 different domain zones (rock salt, potash lens, shale layer)
    nu = [0.15, 0.2, 0.3]  # Poisson ratio, [-]
else:
    ym = 44e9  # Young's modulus, [Pa] for homogeneous case
    nu = 0.3  # Poisson ratio, [-] for homogeneous case

# K = 24.3e9
# ym = 3 * K * (1 - 2 * nu)
# G = 11.2e9
# G = G - 0.5 * G
# nu = (3 * K - 2 * G) / (2 * (3 * K + G))

et = [0]  # elapsed time
filename = 'new_cave2.msh'
mesh = Mesh('./mesh/' + filename, scale, scale)

# modeling heterogeneity
if impurities == True:
    # generates impurities content across the domain and assigns it to every fe
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
fo, _ = V.load_vector(rho, temp, g, depth, pressure, 'cavern')
d_bnd = mesh.extract_bnd(lx=True, ly=False,
                         rx=False, ry=False,
                         bx=False, by=True,
                         tx=False, ty=False)
k, fo = impose_dirichlet(k, fo, d_bnd)

# calculate displacements
u = solve_disp(k, fo)

# calculate strains and stresses
straing = V.gauss_strain(u)
strain = V.nodal_extrapolation(straing)
stressg = V.gauss_stress(straing)
stress = V.nodal_extrapolation(stressg)

# initializing output parameters
dstress = np.zeros((3, mesh.nnodes()))
dstrain_cr = np.zeros((3, mesh.nnodes()))
disp_out = np.zeros((2 * mesh.nnodes(), nt))
stress_out = np.zeros((3 * mesh.nnodes(), nt))
strain_out = np.zeros((3 * mesh.nnodes(), nt))
forces_out = np.zeros((2 * mesh.nnodes(), nt))
svm_out = np.zeros((mesh.nnodes(), nt))
strain_crg = np.zeros((3, mesh.nele()))
strain_cr_out = np.zeros((3 * mesh.nnodes(), nt))

# used to calculate principal stresses if necessary
if principal_stress == True:
    p_stress_out1 = (1 / 2 * (stress[0] + stress[1]) + np.sqrt(
        (1 / 2 * (stress[0] - stress[1])) ** 2 + stress[2] ** 2)).transpose()
    p_stress_out2 = (1 / 2 * (stress[0] + stress[1]) - np.sqrt(
        (1 / 2 * (stress[0] - stress[1])) ** 2 + stress[2] ** 2)).transpose()
    stress_out[:, 0] = np.concatenate((p_stress_out1, p_stress_out2, stress[2]), axis=0)
else:
    stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)

disp_out[:, 0] = np.concatenate((u[::2].reshape(mesh.nnodes(), ), u[1::2].reshape(mesh.nnodes(), )), axis=0)
strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
svm_out[:, 0] = von_mises_stress(stress).transpose()

g_cr = 3 / 2 * a * np.abs(np.power(von_mises_stress(stress), n - 2)) * von_mises_stress(
    stress) * deviatoric_stress(stress) * np.exp(- q / (r * temp))

# damage evolution parameter
if damage == True:
    omega = np.zeros((mesh.nele(),))
    b = bo * np.exp(-q / r / temp)

# cyclic load stress
if cyclic == True:
    pressure = 'max'
    fo, _ = V.load_vector(rho, temp, g, depth, pressure, boundary, sign=1)
    d_bnd = mesh.extract_bnd(lx=True, ly=False,
                             rx=False, ry=False,
                             bx=False, by=True,
                             tx=False, ty=False)
    k, fo = impose_dirichlet(k, fo, d_bnd)
    u = solve_disp(k, fo)

    straing = V.gauss_strain(u)
    strain = V.nodal_extrapolation(straing)
    stressg1 = V.gauss_stress(straing)
    stress1 = V.nodal_extrapolation(stressg)

    pressure = 'min'
    fo, _ = V.load_vector(rho, temp, g, depth, pressure, boundary, sign=-1)
    d_bnd = mesh.extract_bnd(lx=True, ly=False,
                             rx=False, ry=False,
                             bx=False, by=True,
                             tx=False, ty=False)
    k, fo = impose_dirichlet(k, fo, d_bnd)
    u = solve_disp(k, fo)

    straing = V.gauss_strain(u)
    strain = V.nodal_extrapolation(straing)
    stressg2 = V.gauss_stress(straing)
    stress2 = V.nodal_extrapolation(stressg)

# explicit solver
if not NR:
    if nt > 1:
        for i in tqdm(range(nt - 1)):
            if cyclic:
                fo, sign = V.load_vector(rho, temp, g, depth, pressure, boundary, sign, i, 3)
            if damage:
                omega = b * (von_mises_stress(stressg).transpose()) ** n / (((1 - omega) * dt) ** l)
                s_I = 1 / 2 * (stressg[0] - stressg[1]) + 1 / 2 * np.sqrt(
                    (stressg[0] - stressg[1]) ** 2 + 4 * (stressg[2]) ** 2)
                svmg = von_mises_stress(stressg).transpose()
                sw_eq = 1 / 2 * (s_I + svmg)

            if damage:
                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius, omega)
            else:
                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius)

            strain_cr = V.nodal_extrapolation(strain_crg)
            f = fo - f_cr
            f[d_bnd] = 0
            u = solve_disp(k, f)

            # calculate strains
            straing = V.gauss_strain(u)
            strain = V.nodal_extrapolation(straing)

            if cyclic:
                if sign < 0:
                    stress = stress2
                    stressg = stressg2
                else:
                    stress = stress1
                    stressg = stressg1

            # write output data
            disp_out[:, i + 1] = np.concatenate((u[::2].reshape(mesh.nnodes(), ),
                                                 u[1::2].reshape(mesh.nnodes(), )), axis=0)
            strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
            stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
            forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(mesh.nnodes(), ),
                                                   f_cr[1::2].reshape(mesh.nnodes(), )), axis=0)
            svm_out[:, i + 1] = von_mises_stress(stress).transpose()
            strain_cr_out[:, i + 1] = np.concatenate((strain_cr[0], strain_cr[1], strain_cr[2]), axis=0)
            if principal_stress == True:
                p_stress_out1 = (1 / 2 * (stress[0] + stress[1]) + np.sqrt(
                    (1 / 2 * (stress[0] - stress[1])) ** 2 + stress[2] ** 2)).transpose()
                p_stress_out2 = (1 / 2 * (stress[0] + stress[1]) - np.sqrt(
                    (1 / 2 * (stress[0] - stress[1])) ** 2 + stress[2] ** 2)).transpose()
                stress_out[:, i + 1] = np.concatenate((p_stress_out1, p_stress_out2, stress[2]), axis=0)
            else:
                stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)

            if damage and abs(np.max(omega)) >= 0.3:
                disp_out[:, i + 1] = float('nan')
                strain_out[:, i + 1] = float('nan')
                stress_out[:, i + 1] = float('nan')
                forces_out[:, i + 1] = float('nan')
                svm_out[:, i + 1] = float('nan')
                strain_cr_out[:, i + 1] = float('nan')

            # elapsed time
            et = np.append(et, et[-1] + dt)

# implicit solver
if NR:
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
# write_results(nt, mesh, output, filename.split(".")[0], '.eps', amp=True)
# write_results(nt, mesh, output, filename.split(".")[0], '.xdmf')
# save parameters in point A evolution in time
save_plot_A(nt, mesh, output, filename.split(".")[0], 700)
# write results to xls file
# write_xls(filename, output)
