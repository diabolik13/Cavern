import numpy as np
import sympy as sp
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from mpl_toolkits import mplot3d
from scipy import integrate
import sys
import timeit
from test8 import *
from test7 import *

# Timer:
# start_time = timeit.default_timer()

# Cavern Mesh Viscoplasticity
# All metric units used

# mesh_filename = 'rect.msh'            # supported formats: *.mat and *.msh
mesh_filename = 'cave_67.msh'          # supported formats: *.mat and *.msh

# mesh_filename = 'irreg.msh'     # ph_group[i] == 11:
# mesh_filename = '2cavs.msh'     # error in cavern_boundaries function: "list index out of range"

m, p, t = load_mesh(mesh_filename)      # Both numbering (global & local) is done clockwise

# Dimensions:
dof = 2                                 # x and y only
z = 1                                   # thickness in z direction (m)

rho = 2160  # test
g = 9.81    # test
pa = 0.1        # test, atmospheric reference pressure in MPa

# Elastic moduli
Kb = 18115e6                            # Bulk modulus (Pa)
mu = 9842e6                             # Shear modulus (Pa)

# Cavern Pressure (difference lithostatic and cavern pressure):
# pr = -8e6                                                   # pressure (Pa)
# test2 = -np.linspace(2e6, 8e6, 7)                           # was first called test2!       og.. 8e6
# test1 = -np.linspace(7e6, 3e6, 5)
# pr = np.concatenate(4*(test2, test1), axis=0)               # increase/decrease cycle
# pr = np.concatenate(2*(test1, test2), axis=0)               # decrease/increase cycle
# pr = np.insert(pr, len(pr), -2e6)
# pr = np.insert(pr, 0, -1e6)                                    # to find onset of viscoplasticity


# Empty - Full cavern case:     # TODO create a different array of pressures for paper case!
test3 = -np.linspace(8e6, 2e6, 7)
test4 = -np.linspace(3e6, 7e6, 5)
pr = np.concatenate(3*(test3, test4), axis=0)
pr = np.insert(pr, len(pr), -8e6)
pr = np.insert(pr, 0, -7e6)
pr = np.insert(pr, 0, -6e6)
pr = np.insert(pr, 0, -5e6)
pr = np.insert(pr, 0, -4e6)
pr = np.insert(pr, 0, -3e6)
pr = np.insert(pr, 0, -2e6)
pr = np.insert(pr, 0, -1.5e6)       # added this point
pr = np.insert(pr, 0, -1e6)
# pr = np.insert(pr, 0, -9.5e5)       # added this point
pr = np.insert(pr, 0, -9e5)
pr = np.insert(pr, 0, -8e5)
pr = np.insert(pr, 0, -7e5)
pr = np.insert(pr, 0, -6e5)
pr = np.insert(pr, 0, -5e5)
pr = np.insert(pr, 0, -4e5)
pr = np.insert(pr, 0, -3e5)
pr = np.insert(pr, 0, -2e5)
pr = np.insert(pr, 0, -1e5)     # onset of viscoplasticit starts here!
# until here okay:
pr = np.insert(pr, 17, -8e6)
pr = np.insert(pr, 18, -8e6)
pr = np.insert(pr, 19, -8e6)

pr = np.insert(pr, 26, -2e6)
pr = np.insert(pr, 27, -2e6)
pr = np.insert(pr, 28, -2e6)

pr = np.insert(pr, 35, -8e6)
pr = np.insert(pr, 36, -8e6)
pr = np.insert(pr, 37, -8e6)

pr = np.insert(pr, 44, -2e6)
pr = np.insert(pr, 45, -2e6)
pr = np.insert(pr, 46, -2e6)

pr = np.insert(pr, 53, -8e6)
pr = np.insert(pr, 54, -8e6)
pr = np.insert(pr, 55, -8e6)

pr = np.insert(pr, 62, -2e6)
pr = np.insert(pr, 63, -2e6)
pr = np.insert(pr, 64, -2e6)
#
pr = np.insert(pr, 0, -8e6)
# pr = np.insert(pr, 0, -1e3)       # from here on further the pressure gives an error!
# pr = np.insert(pr, 0, -1e1)
# pr = np.insert(pr, 0, 0)

# pr = np.insert(pr, 0, -7e6)

# def cavern_pressure(c):      # might need to adapt this later on
#     inc_pr = -np.linspace(2e6, 8e6, 7)
#     dec_pr = -np.linspace(7e6, 3e6, 5)
#     pr = np.concatenate(c * (inc_pr, dec_pr), axis=0)
#     pr = np.insert(pr, len(pr), -2e6)
#     return pr

# find the elements:
# tosti = np.where(p[1] == 0)
# test = np.where(t == 30)  # find the appropriate element

# Viscoplastic parameters:
sigma_t = 3 * 1.8                       # Tensile strength (MPa)
alpha = np.zeros(len(t[0]))             # (MPa-1)
alpha_q = np.zeros(len(t[0]))           # (MPa-1))

n = 3                                   # (-)
gamma = 0.11                            # (-)
beta_1 = 4.8e-3                         # (MPa-1)
beta = 0.995                            # (-)
mv = -0.5                               # (-)

eta = 0.7                               # (-)
alpha_1 = 0.00005                       # (MPa^(2-n))

mu1 = 5.06e-7 / 86400                   # (s-1)
F0 = 1                                  # (check units!)
N1 = 3                                  # (-)

alpha_0 = 1                             # (MPa)
kv = 0.275                              # (-)

# Switch between solving methods:
implicit = 0                            # solve it one go (0) or with NR (1)
associate = 1                           # associated flow rule (0) or non associated flow rule (1)

# Compression case:
# l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(p, dof)
# d_bnd = np.concatenate((l_bnd, b_bnd))

# Cavern case:
nnodes = len(p[0])
disp_out = np.zeros((2 * nnodes, len(pr)))
stress_out = np.zeros((3 * nnodes, len(pr)))
strain_out = np.zeros((3 * nnodes, len(pr)))
lin_disp_out = np.zeros((2 * nnodes, len(pr)))
lin_stress_out = np.zeros((3 * nnodes, len(pr)))
lin_strain_out = np.zeros((3 * nnodes, len(pr)))
evp_out = np.zeros((3 * nnodes, len(pr)))           # total accumulated viscoplastic strain for plotting purposes
evp_tot = np.zeros((3, len(t[0])))              # array with total accumulated viscoplastic strain
I1_out = np.zeros((len(t[0]), len(pr)))         # for yield function plot
J2_out = np.zeros((len(t[0]), len(pr)))         # for yield function plot
theta_out = np.zeros((len(t[0]), len(pr)))


for j in range(len(pr)):
    l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(p, dof)
    d_bnd = np.concatenate((l_bnd, r_bnd, t_bnd, b_bnd))
    px, py, nind_c = cavern_boundaries(m, p, pr[j], z)

    # Construct K:
    D = elastic_moduli(Kb, mu)
    k = assemble_stiffness_matrix(dof, p, t, D, z)

    # Check K:
    sym_check_K = check_symmetric(k, rtol=1e-05, atol=1e-08)
    sin_check_K = check_singularity(k)

    # Construct F:
    # Compression case (rectangular mesh):
    # nind_c = 0
    # f0 = assemble_vector(p, t, nind_c, px=0, py=0)

    # Cavern mesh case:
    f0 = assemble_vector(p, t, rho, g, nind_c, px, py)

    # Impose Boundary Conditions:
    k, f0 = impose_dirichlet(k, f0, d_bnd)

    # Solve the system
    u = np.linalg.solve(k, f0)

    # Check the system
    check = np.allclose(np.dot(k, u), f0, rtol=1e-4, atol=1e-4)  # set the tolerance
    residual = np.dot(k, u) - f0

    # Remember CST formulation for both stress and strain:
    straing, stressg = gauss_stress_strain(p, t, u, D)
    strain, stress = nodal_stress_strain(p, t, straing, stressg)

    lin_disp_out[:, j] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
    lin_strain_out[:, j] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
    lin_stress_out[:, j] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)

    # Plot Results:
    plot_results(p, t, 'displacement', 'strain', 'stress', u, strain, stress)
    # negative displacement means with orientation of axis <-
    # positive displacement means with orientation of axis ->
    # negative strain: compression (cavern shrinks)
    # positive strain: (ex)tension (cavern grows)
    # negative stress: exert stress into the cavern
    # positive stress: exert stress outside the cavern

    # Check for pressure in both directions
    #fx = f0[::2]
    #fy = f0[1::2]

    # plot_parameter(p, t, stress[0, :])
    # deformed_mesh(p, t, u)

    if implicit == 0:
        stress_mpa = convert_stress(stressg)

        evp = np.zeros((3, len(t[0])))  # used for the alpha coefficient, I could remove this one
        Fvp = np.zeros(len(t[0]))  # used for the alpha coefficient
        I1 = first_stress_inv(stress_mpa, sigma_t)
        I2 = second_stress_inv(stress_mpa)

        J2 = second_dev_stress_inv(I1, I2)
        J3 = third_dev_stress_inv(I1, I2)
        theta, theta_degrees = lode_angle(stress_mpa, J2, J3)

        alpha, alpha_q = hardening_param(t, straing, Fvp, alpha_1, eta, alpha_0, kv, associate)
        Fvp = yield_function(I1, J2, alpha, theta, beta_1, beta, mv, n, gamma, F0, pa)        # o.g.
        check_Fvp = Fvp[Fvp > 0]            # @2e6 Pa, 880 elements are above threshold (currently)

        # Plot yield function at nodal points over the entire mesh:
        # Fvp_nod = nodal_yield_function(p, t, Fvp)
        # plot_parameter(p, t, Fvp_nod)

        Qvp = potential_function(I1, J2, alpha_q, theta, beta_1, beta, mv, n, gamma, pa)

        # Potential function derivatives:
        dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy, dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy = stress_inv_der(stress_mpa[0, :],
                                                                                                stress_mpa[1, :],
                                                                                                stress_mpa[2, :],
                                                                                                sigma_t)
        dQvpdI1, dQvpdJ2, dQvpdJ3 = pot_der(alpha_q, beta_1, beta, mv, n, gamma, I1, J2, J3, pa)
        dQvpdxx, dQvpdyy, dQvpdxy = chain_rule(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy,
                                               dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy)

        evp = viscoplastic_strain(t, Fvp, dQvpdxx, dQvpdyy, dQvpdxy, mu1, N1, 5000)       # scaled this one for first cycle to 50 (o.g. 1)
        evp_tot = evp_tot + evp
        fvp = assemble_vp_force_vector(dof, p, t, D, evp_tot, z)

        f = f0 - fvp
        k, f = impose_dirichlet(k, f, d_bnd)
        u1 = np.linalg.solve(k, f)                                      # explicit solution

        straing1, stressg1 = gauss_stress_strain(p, t, u1, D)
        strain1, stress1 = nodal_stress_strain(p, t, straing1, stressg1)
        evp_nod, _ = nodal_stress_strain(p, t, evp_tot, stressg)        # accumulated visco strain
        plot_results(p, t, 'displacement', 'strain', 'stress', u1, strain1, stress1)

        # plot_parameter(p, t, evp_nod[0, :])  # for x direction
        # plot_parameter(p, t, evp_nod[1, :])
        disp_out[:, j] = np.concatenate((u1[::2].reshape(nnodes, ), u1[1::2].reshape(nnodes, )), axis=0)        # total displacement
        strain_out[:, j] = np.concatenate((strain1[0], strain1[1], strain1[2]), axis=0)                         # total strain
        stress_out[:, j] = np.concatenate((stress1[0], stress1[1], stress1[2]), axis=0)                         # total stress
        evp_out[:, j] = np.concatenate((evp_nod[0], evp_nod[1], evp_nod[2]), axis=0)                            # total viscoplastic strain
        I1_out[:, j] = I1
        J2_out[:, j] = J2
        theta_out[:, j] = theta_degrees

    else:
        max_iter = 10
        converged = 0
        iter = 0
        conv = 1e-4  # tolerance

        evp = np.zeros((3, len(t[0])))                      # used for the alpha coefficient
        Fvp = np.zeros(len(t[0]))                                 # used for the alpha coefficient

        straing, stressg = gauss_stress_strain_tot(p, t, u, D, evp_tot)     # total strain
        strain, stress = nodal_stress_strain(p, t, straing, stressg)

        while converged == 0:
            stress_mpa = convert_stress(stressg)
            I1 = first_stress_inv(stress_mpa, sigma_t)

            I2 = second_stress_inv(stress_mpa)
            J2 = second_dev_stress_inv(I1, I2)
            J3 = third_dev_stress_inv(I1, I2)

            theta, theta_degrees = lode_angle(stress_mpa, J2, J3)
            alpha, alpha_q = hardening_param(t, straing, Fvp, alpha_1, eta, alpha_0, kv, associate)
            Fvp = yield_function(I1, J2, alpha, theta, beta_1, beta, mv, n, gamma, F0, pa)

            Qvp = potential_function(I1, J2, alpha_q, theta, beta_1, beta, mv, n, gamma, pa)

            # Potential function derivatives:
            dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy, dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy = stress_inv_der(stress_mpa[0, :],
                                                                                                    stress_mpa[1, :],
                                                                                                    stress_mpa[2, :],
                                                                                                    sigma_t)
            dQvpdI1, dQvpdJ2, dQvpdJ3 = pot_der(alpha_q, beta_1, beta, mv, n, gamma, I1, J2, J3, pa)
            dQvpdxx, dQvpdyy, dQvpdxy = chain_rule(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy,
                                                   dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy)

            evp = viscoplastic_strain(t, Fvp, dQvpdxx, dQvpdyy, dQvpdxy, mu1, N1, 1)

            fvp = assemble_vp_force_vector(dof, p, t, D, evp, z)
            f = f0 - fvp        # changed sign from + to -
            k, f = impose_dirichlet(k, f, d_bnd)

            residual = np.dot(k, u) - f
            delta_u = -np.linalg.solve(k, residual)

            u = u + delta_u

            # update properties
            straing, stressg = gauss_stress_strain(p, t, u, D)
            strain, stress = nodal_stress_strain(p, t, straing, stressg)

            stress_mpa = convert_stress(stressg)

            I1 = first_stress_inv(stress_mpa, sigma_t)
            I2 = second_stress_inv(stress_mpa)
            J2 = second_dev_stress_inv(I1, I2)
            J3 = third_dev_stress_inv(I1, I2)

            theta, theta_degrees = lode_angle(stress_mpa, J2, J3)
            alpha, alpha_q = hardening_param(t, straing, Fvp, alpha_1, eta, alpha_0, kv, associate)

            Fvp = yield_function(I1, J2, alpha, theta, beta_1, beta, mv, n, gamma, F0, pa)

            Qvp = potential_function(I1, J2, alpha_q, theta, beta_1, beta, mv, n, gamma, pa)

            # Potential function derivatives:
            dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy, dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy = stress_inv_der(stress_mpa[0, :],
                                                                                                    stress_mpa[1, :],
                                                                                                    stress_mpa[2, :],
                                                                                                    sigma_t)
            dQvpdI1, dQvpdJ2, dQvpdJ3 = pot_der(alpha_q, beta_1, beta, mv, n, gamma, I1, J2, J3, pa)
            dQvpdxx, dQvpdyy, dQvpdxy = chain_rule(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy,
                                                   dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy)

            evp = viscoplastic_strain(t, Fvp, dQvpdxx, dQvpdyy, dQvpdxy, mu1, N1, 1)

            fvp = assemble_vp_force_vector(dof, p, t, D, evp, z)

            f = f0 - fvp    # changed sign from + to -
            k, f = impose_dirichlet(k, f, d_bnd)

            # re-compute residual
            residual = np.dot(k, u) - f
            res = np.linalg.norm(residual)
            iter += 1

            print("Iteration {}, norm(residual) = {}.".format(iter, res))
            if iter == max_iter and res >= conv:
                print("Maximum iterations reached.")
            if res < conv or iter >= max_iter:
                converged = 1

        # Plot yield function at nodal points over the entire mesh:
        # Fvp_nod = nodal_yield_function(p, t, Fvp)
        # plot_parameter(p, t, Fvp_nod)

        evp_tot = evp_tot + evp
        evp_nod, _ = nodal_stress_strain(p, t, evp_tot, stressg)
        straing, stressg = gauss_stress_strain_tot(p, t, u, D, evp_tot)  # total strain     Kishand told me there is no need for this
        strain, stress = nodal_stress_strain(p, t, straing, stressg)                    # Kishan told me there is no need for this
        disp_out[:, j] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
        strain_out[:, j] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
        stress_out[:, j] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
        evp_out[:, j] = np.concatenate((evp_nod[0], evp_nod[1], evp_nod[2]), axis=0)
        I1_out[:, j] = I1
        J2_out[:, j] = J2

        # plot_results(p, t, 'displacement', 'strain', 'stress', u, strain, stress)


# plot_results(p, t, 'displacement', 'strain', 'stress', u, strain, stress)

# plot_results(p, t, 'displacement', 'strain', 'stress', u1, strain1, stress1)        # explicit solution looks now fine for 1 cycle (magnified!)
plot_results(p, t, 'displacement', 'strain', 'stress', u, strain, stress)         # implicit solution, 2 cycles check how it looks!


Fvp_nod = nodal_yield_function(p, t, Fvp)
plot_parameter(p, t, Fvp_nod)

# Time property (assuming 1 MPa increase/decrease per day AND constant injection/depletion:
t_end = len(pr)
t = np.linspace(0, t_end, len(pr))      # was len(pr) + 1
# gives the ability to plot evp_tot vs time

# find the elements:
# test = np.where(t == 91)  # find the appropriate element

# plt.figure()
# plt.plot(evp_out[5, :], pr, '-ob')
# plt.plot(strain_out[5, :], pr, '-og')
# plt.plot((strain_out[5, :] - evp_out[5, :]), pr, '-or')
# plt.plot(lin_strain_out[5, :], pr, '-om')
# plt.plot((evp_out[5, :] + lin_strain_out[5, :]), pr, '-oy')
# plt.legend(('Viscoplastic strain', 'Total strain', 'Linear elastic strain (deduction)', 'Linear elastic strain (model)', 'Total strain (addition)'))
# plt.show()

# quick overview if evp is really nonlinear
# plt.figure()        # looks alright :)
# plt.plot(evp_out[5, :], pr)
# plt.show()

# quick overview if lin el is a straight line
# plt.figure()        # looks alright :)
# plt.plot(strain_out[5, :] - evp_out[5, :], pr)
# plt.show()

# Yield function plot: (element 3)
# alpha_t = np.array([0, 0.00024, 0.0006, 0.00093, 0.0018])
# I1a = np.linspace(0, 120, 50)
# I1b = I1a + sigma_t  # MPa
# J2a = np.zeros((5, 50))
# for i in range(len(alpha_t)):
#     J2a[i] = (-alpha_t[i] * np.power(I1b, n) + gamma * np.power(I1b, 2)) * np.power(
#         np.exp(beta_1 * I1b) - beta * np.cos(3 * theta[0]), mv)
#     J2a[J2a < 0] = 0

# plt.figure()  # could add the dilatancy boundary in the figure
# plt.plot(I1a, np.sqrt(J2a[0, :]), '-o')  # line plots
# plt.plot(I1a, np.sqrt(J2a[1, :]), '-o')
# plt.plot(I1a, np.sqrt(J2a[2, :]), '-o')
# plt.plot(I1a, np.sqrt(J2a[3, :]), '-o')
# plt.plot(I1a, np.sqrt(J2a[4, :]), '-o')
# plt.plot(I1_out[3, :], np.sqrt(J2_out[3, :]), 'ob')
# plt.plot(I1_out[9, :], np.sqrt(J2_out[9, :]), 'og')
# plt.xlabel('I1 ')
# plt.ylabel('sqrt(J2)')
# plt.ylim(0, 45)
# plt.title('Yield function')
# plt.show()

# plt.figure()
# plt.plot(np.abs(strain_out[566, :]), np.abs(pr), '-ob')
# plt.plot(np.abs(strain_out[487, :]), np.abs(pr), '-og')
# plt.plot(np.abs(strain_out[455, :]), np.abs(pr), '-om')
# plt.plot(strain_out[566, :], pr, '-ob')
# plt.plot(strain_out[487, :], pr, '-og')
# plt.plot(strain_out[455, :], pr, '-om')
# plt.xlabel('Strain (-)')
# plt.ylabel('Pressure (Pa)')
# plt.title('5 Loading/unloading cycles for element 5 in x direction')
# plt.title('Loading unloading cycle for element 566 in x direction')
# plt.legend(('Far from cavern', 'Middle of cavern ', 'Far from cavern'))
# plt.show()

# plot_results(p, t, 'displacement', 'strain', 'stress', disp_out[:, 0], strain_out[:, 0], stress_out[:, 0])       # 0
# plot_results(p, t, 'displacement', 'strain', 'stress', u, strain, stress)       # 6 or increase the pressure further dependent on the image!


# plot of total strain over entire mesh over x cycles:
# plt.figure()
# plot_parameter(p, t, strain1[0])




# plot for time vs strain for 1 year
plt.figure()
plt.plot(t, np.abs(strain_out[6, :]), 'b', label='element 6')
plt.plot(t, np.abs(strain_out[23, :]), 'r', label='element 23')
plt.xlabel('Time (days)')
plt.ylabel('Strain (-)')
plt.title('3 loading cycles in x direction')
plt.legend(loc='lower right')
plt.show()

# plot for strain vs pressure for one cycle
plt.figure()
plt.plot(np.abs(strain_out[5, :]), np.abs(pr))
plt.xlabel('Strain (-)')
plt.ylabel('Pressure (Pa)')
plt.title('4 Loading unloading cycle for element 5 in x direction')
plt.show()

plt.figure()
_ = plt.hist(evp[0, :], bins=5)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.show()

# code you want to evaluate
# elapsed = timeit.default_timer() - start_time

print('done')
