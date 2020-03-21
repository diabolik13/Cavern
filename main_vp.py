from elasticity import *
from viscoplasticity import *
from animate_plot import *
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import pyplot

import matplotlib.pyplot as pyplt
import sympy as sp

mesh_filename = 'rect.msh'  # supported formats: *.mat and *.msh
input_param = load_input(mesh_filename)
k = input_param['stiffness matrix']
# f = input_param['external forces']
d = input_param['elasticity tensor']
p = input_param['points']
t = input_param['elements']
th = input_param['thickness']
lamda = input_param['Lame parameter']
mu = input_param['shear moduli']
d_bnd = input_param['Dirichlet boundaries']
nnodes = len(p[0])
x = p[0]
y = p[1]

# u = np.linalg.solve(k, f)
# straing, stressg = gauss_stress_strain(p, t, u, d)
# strain, stress = nodal_stress_strain(p, t, straing, stressg)

xsym, ysym = sp.symbols('xsym ysym')
# c1 = 0.5
# c2 = - np.pi / 2
c1 = 1
c2 = 0
u_x = 1e1 * xsym * sp.sin(c1 * np.pi * xsym / 1000 + c2) * sp.sin(c1 * np.pi * ysym / 1000 + c2)
u_y = -1e1 * ysym * sp.sin(c1 * np.pi * xsym / 1000 + c2) * sp.sin(c1 * np.pi * ysym / 1000 + c2)

# derivatives, rquired to calculate strain, stress and forces analytically
ux_anl = sp.lambdify([xsym, ysym], u_x, "numpy")(x, y)
uy_anl = sp.lambdify([xsym, ysym], u_y, "numpy")(x, y)
dudx = sp.lambdify([xsym, ysym], sp.diff(u_x, xsym), "numpy")(x, y)
dvdy = sp.lambdify([xsym, ysym], sp.diff(u_y, ysym), "numpy")(x, y)
dudy = sp.lambdify([xsym, ysym], sp.diff(u_x, ysym), "numpy")(x, y)
dvdx = sp.lambdify([xsym, ysym], sp.diff(u_y, xsym), "numpy")(x, y)
du2dx = sp.lambdify([xsym, ysym], sp.diff(u_x, xsym, 2), "numpy")(x, y)
dv2dx = sp.lambdify([xsym, ysym], sp.diff(u_y, xsym, 2), "numpy")(x, y)
du2dy = sp.lambdify([xsym, ysym], sp.diff(u_x, ysym, 2), "numpy")(x, y)
dv2dy = sp.lambdify([xsym, ysym], sp.diff(u_y, ysym, 2), "numpy")(x, y)
du2dxdy = sp.lambdify([xsym, ysym], sp.diff(u_x, xsym, ysym), "numpy")(x, y)
dv2dxdy = sp.lambdify([xsym, ysym], sp.diff(u_y, xsym, ysym), "numpy")(x, y)

if np.isscalar(dudx):
    dudx = np.zeros((nnodes,))
if np.isscalar(dvdy):
    dvdy = np.zeros((nnodes,))
if np.isscalar(dudy):
    dudy = np.zeros((nnodes,))
if np.isscalar(dvdx):
    dvdx = np.zeros((nnodes,))
if np.isscalar(du2dx):
    du2dx = np.zeros((nnodes,))
if np.isscalar(dv2dx):
    dv2dx = np.zeros((nnodes,))
if np.isscalar(du2dy):
    du2dy = np.zeros((nnodes,))
if np.isscalar(dv2dy):
    dv2dy = np.zeros((nnodes,))
if np.isscalar(du2dxdy):
    du2dxdy = np.zeros((nnodes,))
if np.isscalar(dv2dxdy):
    dv2dxdy = np.zeros((nnodes,))

strain_anl = np.concatenate((dudx, dvdy, dudy + dvdx), axis=0).reshape((3 * len(p[0]), 1))
sx = (lamda + 2 * mu) * dudx + lamda * dvdy
sy = (lamda + 2 * mu) * dvdy + lamda * dudx
ss = mu * (dudy + dvdx)
stress_anl = np.concatenate((sx, sy, ss), axis=0).reshape((3 * len(p[0]), 1))

fx = -((lamda + 2 * mu) * du2dx + lamda * dv2dxdy + mu * (du2dy + dv2dxdy))
fy = -((lamda + 2 * mu) * dv2dy + lamda * du2dxdy + mu * (dv2dx + du2dxdy))
f = assemble_vector(p, t, th, fx, fy)
k, f = impose_dirichlet(k, f, d_bnd)

u = np.linalg.solve(k, f)
straing, stressg = gauss_stress_strain(p, t, u, d)
strain, stress = nodal_stress_strain(p, t, straing, stressg)
ux_num = u[::2]
uy_num = u[1::2]
u = np.concatenate((ux_num, uy_num), axis=1).transpose()

u_anl = np.concatenate((ux_anl.reshape(nnodes, 1), uy_anl.reshape(nnodes, 1)), axis=1).transpose()
strain_anl = np.concatenate((dudx.reshape(nnodes, 1), dvdy.reshape(nnodes, 1), (dudy + dvdx).reshape(nnodes, 1)),
                            axis=1).transpose()
stress_anl = np.concatenate((sx.reshape(nnodes, 1), sy.reshape(nnodes, 1), ss.reshape(nnodes, 1)), axis=1).transpose()

plot_results(p, t, 'disp', 'strain', 'stress', abs(u_anl - u), abs(strain_anl - strain), abs(stress_anl - stress))
plot_results(p, t, 'disp', 'strain', 'stress', u, strain, stress)
print('Maximum absolute value of dispacement is ' + str(abs(np.max(u))) + ' meters.')

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

# Solve for Stress invariants:
# I1, J2, J3 = stress_inv(stressg)
I1, J2, J3 = stress_inv(stress)

# Lode Angle:
theta, theta_degrees, avg_theta = lode_angle(J2, J3)

# Yield Function:
Fvp = solve_yield_function(I1, J2, alpha, n, gamma, beta, beta_1, m_v, theta)
# Fvp = np.nan_to_num(Fvp)

# Create ultimate failure boundary line;
J2_UF = (-alpha * np.power(I1, n) + gamma * np.power(I1, 2)) * np.power(
    (np.exp(beta_1 * I1) - beta * np.cos(3 * theta)), m_v)
I1_test = np.linspace(0, 120, 50)
J2_UF_Test = (-alpha * np.power(I1_test, n) + gamma * np.power(I1_test, 2)) * np.power(
    (np.exp(beta_1 * I1_test) + beta), m_v)

fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(fx, fy, Fvp)
ax.set_xlabel('displacement in X')
ax.set_ylabel('displacement in Y')
ax.set_zlabel('Fvp')
pyplot.show()

pyplt.figure(1)
pyplt.plot(I1, np.sqrt(J2), 'o')
pyplt.plot(I1, np.sqrt(J2_UF), '-o')
pyplt.xlabel('I1 (MPa)')
pyplt.ylabel('sqrt(J2) (MPa)')
pyplt.title('Fvp evaluation at each nodal point')
# pyplt.show()

pyplt.figure(2)
pyplt.plot(I1, np.sqrt(J2), 'o')
pyplt.plot(I1_test, np.sqrt(J2_UF_Test), '-o')
pyplt.xlabel('I1 (MPa)')
pyplt.ylabel('sqrt(J2) (MPa)')
pyplt.title('Viscoplastic yield function with theta = 60 degrees')
# pyplt.show()

# Derivatives of potential function with respect to stresses:
dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy = potential_function_chain(alpha_q, n, gamma, beta_1, beta, m_v, I1, J2, J3,
                                                                    sigma_t, stressg)

# Solve for viscoplastic formulation (Desai):
evp = desai(mu1, Fvp, F0, N1, dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy)

# Create viscoplastic force:
fvp = assemble_vp_force_vector(2, p, t, d, evp, th)

test = f - fvp

# Building NR solver
converged = 0
iter = 0
max_iter = 10
epsilon = 1e-5

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

# TODO residual function (res) plot it with respect to different parameters
print('Done')
