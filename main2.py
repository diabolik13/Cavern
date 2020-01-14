import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from scipy.io import loadmat
from matplotlib import cm
from numpy import inf

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def extract_bnd(p, nnodes, dof):
    l_bnd = np.array([], dtype='i')
    r_bnd = np.array([], dtype='i')
    b_bnd = np.array([], dtype='i')
    t_bnd = np.array([], dtype='i')

    for node in range(nnodes):
        if p[0][node] == -1:
            l_bnd = np.append(l_bnd, node)
        if p[1][node] == -1:
            b_bnd = np.append(b_bnd, node)
        if p[0][node] == 1:
            r_bnd = np.append(r_bnd, node)
        if p[1][node] == 1:
            t_bnd = np.append(t_bnd, node)

    l_bnd = l_bnd * dof
    r_bnd = r_bnd * dof
    b_bnd = b_bnd * dof + 1
    t_bnd = t_bnd * dof + 1

    return l_bnd, r_bnd, b_bnd, t_bnd


def generate_stiffness_matrix(el, lamda, mu):
    X = np.array(el[:, 0])
    Y = np.array(el[:, 1])
    area = PolyArea(X, Y)
    x = np.zeros((3, 3))
    y = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            x[i, j] = X[i] - X[j]
            y[i, j] = Y[i] - Y[j]

    J = [[x[0, 2], y[0, 2]],
         [x[1, 2], y[1, 2]]]
    B = 1 / np.linalg.det(J) * np.array([[y[1, 2], 0, y[2, 0], 0, y[0, 1], 0],
                                         [0, x[2, 1], 0, x[0, 2], 0, x[1, 0]],
                                         [x[2, 1], y[1, 2], x[0, 2], y[2, 0], x[1, 0], y[0, 1]]])
    D = np.array([[lamda + 2 * mu, lamda, 0],
                  [lamda, lamda + 2 * mu, 0],
                  [0, 0, mu]])
    area2 = 1 / 2 * abs(np.linalg.det(J))
    ke = area * np.dot((np.dot(B.transpose(), D)), B)

    return ke


def assemble_stiffness_matrix(dof, nnodes, p, t, lamda, mu):
    K = np.zeros((dof * nnodes, dof * nnodes))

    for k in range(len(t[0])):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        I = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        ke = generate_stiffness_matrix(el, lamda, mu)

        for i in range(6):
            for j in range(6):
                K[I[i], I[j]] = K[I[i], I[j]] + ke[i, j]

    return K


def assemble_vector(nnodes, p, t):
    f = np.zeros((2 * nnodes, 1))

    for k in range(len(t[0])):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        I = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        fe = np.zeros(6)
        # f_ek = generate_element_vector(el, p)
        j = 0
        for i in range(3):
            X = np.array(el[:, 0])
            Y = np.array(el[:, 1])
            if X[i] == 1:
                fe[j] = -1
            j = j + 2

        for i in range(6):
            f[I[i]] = f[I[i]] + fe[i]

    return f


rho = 2980  # rock density
K = 56.1e9  # Bulk modulus
mu = 29.1e9  # Shear modulus
g = 9.81  # gravity constant
H = 1250  # depth of the middle of the salt layer
dt = 15  # timestep
Nt = 50  # number of timesteps
A = 1e-14  # material constant (Norton Power Law)
n = 3  # stress exponent (Norton Power Law)
P = 1  # cavern's pressure

lamda = K - 2 / 3 * mu  # Elastic modulus
E = mu * (3 * lamda + 2 * mu) / (lamda + mu)  # Young's modulus
nu = lamda / (2 * (lamda + mu))  # Poisson's ratio

mesh = loadmat('rounded_cave3.mat')
# mesh = loadmat('R1.mat')

p = mesh['p']
e = mesh['e']
t = mesh['t']
t = t - 1  # update elements numbering to start with 0
dof = 2
nnodes = p.shape[1]
nele = len(t[0])
x = p[0, :]
y = p[1, :]

L_bnd, R_bnd, B_bnd, T_bnd = extract_bnd(p, nnodes, dof)
I_bnd = np.concatenate((L_bnd, R_bnd, B_bnd, T_bnd))
k = assemble_stiffness_matrix(dof, nnodes, p, t, lamda, mu)
f = assemble_vector(nnodes, p, t)

# Impose Dirichlet B.C.
bnd = np.concatenate((B_bnd, T_bnd, L_bnd))
k[bnd, :] = 0
k[:, bnd] = 0
k[bnd, bnd] = 1
# f[bnd] = 0

u = np.linalg.solve(k, f)

# nz_k = np.count_nonzero(k)
# t1 = "Number of nonzero values in s11 = {}".format(nz_k)
# plt.spy(k)
# plt.title(t1)
# plt.show()

ux = u[::2].reshape(nnodes, )
uy = u[1::2].reshape(nnodes, )

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, ux, linewidth=0.2, antialiased=True, cmap=cm.jet)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(x, y, uy, linewidth=0.2, antialiased=True, cmap=cm.jet)
# plt.show()

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))

# ===============
#  First subplot
# ===============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface
surf = ax.plot_trisurf(x, y, ux, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, shrink=0.5, aspect=10)

# ===============
# Second subplot
# ===============
# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

# plot a 3D surface
surf = ax.plot_trisurf(x, y, uy, cmap=cm.jet,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.view_init(elev=90, azim=-90)
fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()

print("done")
