from elasticity import *
from viscoplasticity import *
from animate_plot import *

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

print('Maximum absolute value of dispacement is ' + str(abs(np.max(u))) + ' meters.')

# Convert stresses from Pa to MPa
stress = stress * 1e-6  # Nodal point
stressg = stressg * 1e-6  # Gaussian point

# Constants to solve every function for viscoplastic behaviour:
mu1 = 5.06e-7  # day-1
F0 = 1  # MPa
N1 = 3  # (-)
alpha = 8e-4  # guessed value!
alpha_q = 8e-4  # guessed value!
n = 3  # (-)
gamma = 0.11  # (-)
beta = 0.995  # (-)
beta_1 = 4.8e-3  # MPa-1
m_v = -0.5  # (-)
th = 1e3  # thickness of the model

# Solve for Stress invariants:
I1, J2, J3 = stress_inv(stressg)

# Solve yield function:
Fvp = solve_yield_function(I1, J2, J3, alpha, n, gamma, beta, beta_1, m_v)

# Derivatives of potential function with respect to stress invariants:
dQvpdI1, dQvpdJ2, dQvpdJ3 = pot_derivatives(stressg, alpha_q, n, gamma, beta_1, m_v, beta, I1, J2, J3)

# Derivatives of stress invariants with respect to stress direction:
dI1dsigmaxx, dJ2dsigmaxx, dJ3dsigmaxx, dI1dsigmayy, dJ2dsigmayy, dJ3dsigmayy, dI1dsigmaxy, dJ2dsigmaxy, dJ3dsigmaxy = \
    der_stress_inv(t, stressg)

# Chain rule for the potential function:
dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy = potential_stress(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dsigmaxx, dI1dsigmayy,
                                                            dI1dsigmaxy, dJ2dsigmaxx, dJ2dsigmayy, dJ2dsigmaxy,
                                                            dJ3dsigmaxx, dJ3dsigmayy, dJ3dsigmaxy)

# Solve for viscoplastic formulation:
evp = desai(mu1, Fvp, F0, N1, dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy)

# Create viscoplastic force:
fvp = assemble_vp_force_vector(2, p, t, d, evp, th)

# Building NR solver
converged = 0
iter = 0
max_iter = 10
epsilon = 1e-5

# Does not use this part below?
while iter < max_iter and converged == 0:
    residual = np.dot(k, u) - f - fvp
    delta_u = - np.linalg.solve(k, residual)

    u = u + delta_u

    # Update residual function
    residual = np.dot(k, u) - f - fvp
    res = np.linalg.norm(residual)
    iter += 1
    if res < epsilon or iter >= max_iter:
        converged = 1
    print('Iteration {}'.format(iter))

# TODO change if statement to all.res < epsilon as each residual function for each element should converge
# TODO make a statement that when Fvp < 0 it is set to zero
# TODO residual function (res) plot it with respect to different parameters
print('Done')
