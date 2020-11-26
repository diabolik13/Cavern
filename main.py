# for paraview post processing: coords + (displacement_x, [m]*iHat + displacement_y, [m]*jHat + 0*kHat)
import time, sys, getopt

from animate_plot import write_results
from classeslib import *
from datetime import datetime
from elasticity import impose_dirichlet, solve_disp, deviatoric_stress, save_plot_A, write_xls
from tqdm import tqdm
from input import *

sys.stdout = Logger("log.txt")
time_start = time.time()

# define shape functions, stiffness matrix, load vector and boundary conditions
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
                # omega = b * (von_mises_stress(stressg).transpose()) ** kk / (((1 - omega) * dt) ** l)
                omega = b * (von_mises_stress(stressg).transpose()) ** kk / ((1 - omega) ** l) * dt
                f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg, strain_crg, arrhenius, omega)
                test = 1
                # s_I = 1 / 2 * (stressg[0] - stressg[1]) + 1 / 2 * np.sqrt(
                #     (stressg[0] - stressg[1]) ** 2 + 4 * (stressg[2]) ** 2)
                # svmg = von_mises_stress(stressg).transpose()
                # sw_eq = 1 / 2 * (s_I + svmg)
                # sw_eq = svmg

            if cyclic:
                if sign > 0:
                    f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg1, strain_crg, arrhenius)
                else:
                    f_cr, strain_crg = V.creep_load_vector(dt, a, n, q, r, temp, stressg2, strain_crg, arrhenius)

            # f_cr = -f_cr
            strain_cr = V.nodal_extrapolation(strain_crg)
            f = fo + f_cr
            f[d_bnd] = 0
            u = solve_disp(k, f)

            # calculate strains
            straing = V.gauss_strain(u)
            strain = V.nodal_extrapolation(straing)

            if cyclic:
                if sign > 0:
                    stress = stress1
                    stressg = stressg1
                else:
                    stress = stress2
                    stressg = stressg2

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
# write_results(nt, mesh, output, filename.split(".")[0], '.eps', amp=False)
# write_results(nt, mesh, output, filename.split(".")[0], '.xdmf')
# save parameters in point A evolution in time
# index = np.where(output['displacement'] == np.amax(output['displacement']))[0]
save_plot_A(nt, mesh, output, filename.split(".")[0], 85, 'png')
# write results to xls file
# write_xls(filename, output)
