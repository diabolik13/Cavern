from elasticity import *
from viscoplasticity import *
from animate_plot import *
import matplotlib.pyplot as pyplt

# this file is used as scratch file
# Objective: find out how nonlinear my viscoplasticity is.

mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
input_param = load_input(mesh_filename)
k = input_param['stiffness matrix']
f = input_param['external forces']
d = input_param['elasticity tensor']
p = input_param['points']
t = input_param['elements']

u = np.linalg.solve(k, f)
straing, stressg = gauss_stress_strain(p, t, u, d)
strain, stress = nodal_stress_strain(p, t, straing, stressg)

# Convert stresses from Pa to MPa
stress = stress * 1e-6  # Nodal point
stressg = stressg * 1e-6  # Gaussian point

# Constants to solve every function for viscoplastic behaviour:
mu1 = 5.06e-7  # day-1
F0 = 1  # MPa
N1 = 3  # (-)
alpha = 0  # guessed value!, 8e-4 (OG)
alpha_q = 8e-4  # guessed value!
n = 3  # (-)
gamma = 0.11  # (-)
beta = 0.995  # (-)
beta_1 = 4.8e-3  # MPa-1
m_v = -0.5  # (-)
th = 1e3  # thickness of the model
sigma_t = 1.8  # MPa, tensile strenght of rock salt
c = 1e-2  # Randomised constant; 0.001
Nt = 10  # number of timesteps

lengte = len(u)
traing = t.shape[1]
disp_u = np.zeros((lengte, Nt))
disp_evp = np.zeros((Nt, 3 * traing))
disp_stressg = np.zeros((Nt, 3 * traing))


for i in range(Nt):
    I1, J2, J3 = stress_inv(stressg)
    theta, theta_degrees, avg_theta = lode_angle(J2, J3)
    Fvp = solve_yield_function(I1, J2, alpha, n, gamma, beta, beta_1, m_v, theta)
    dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy = potential_function_chain(alpha_q, n, gamma, beta_1, beta, m_v, I1, J2,
                                                                        J3,
                                                                        sigma_t, stressg)
    evp = desai(mu1, Fvp, F0, N1, dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy)
    disp_evp[i, :] = np.concatenate((evp[0], evp[1], evp[2]))  # Matrix containing all the results of evp (xx, yy, xy)
    u = u + c
    disp_u[:, i] = np.concatenate(u)  # Matrix containing all the results of u
    straing, stressg = gauss_stress_strain(p, t, u, d)
    strain, stress = nodal_stress_strain(p, t, straing, stressg)
    stress = stress * 1e-6  # Nodal point
    stressg = stressg * 1e-6  # Gaussian point
    disp_stressg[i, :] = np.concatenate(stressg)  # Does not vary

pyplt.figure(1)
pyplt.plot(disp_u[0, :], disp_evp[:, 0], 'o')
pyplt.xlabel('u ')
pyplt.ylabel('evp')
pyplt.title('Nonlinearity check')
pyplt.show()
print('done')