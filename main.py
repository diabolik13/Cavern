import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.io import loadmat


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def extract_bnd(p, nnodes):
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

    return l_bnd, r_bnd, b_bnd, t_bnd


def generate_stiffness_matrix(el1, el2, el3, lamda, mu):
    X = [el1[0], el2[0], el3[0]]
    Y = [el1[1], el2[1], el3[1]]
    area = PolyArea(X, Y)
    m = [[el1[0], el1[1], 1],
         [el2[0], el2[1], 1],
         [el3[0], el3[1], 1]]
    c = np.linalg.solve(m, np.identity(3))

    s_ek11 = np.zeros((3, 3))
    s_ek12 = np.zeros((3, 3))
    s_ek21 = np.zeros((3, 3))
    s_ek22 = np.zeros((3, 3))
    dx_e = np.zeros((3, 3))
    dy_e = np.zeros((3, 3))
    Nk = np.zeros(3)
    # Nl = np.zeros(3)

    # for i in range(3):
    #     Nk[i] = c[0, i] * X[i] + c[1, i] * Y[i] + c[2, i]
    #     for j in range(3):
    #         # Nl[j] = c[0, j] * X[j] + c[1, j] * Y[j] + c[2, j]
    #         s_ek11[i, j] = area * ((lamda + 2 * mu) * c[0, i] * c[0, j] + mu * c[1, i] * c[1, j])
    #         s_ek12[i, j] = area * (lamda * c[1, j] * c[0, i] + mu * c[0, j] * c[1, i])
    #         s_ek22[i, j] = area * ((lamda + 2 * mu) * c[1, i] * c[1, j] + mu * c[0, i] * c[0, j])
    #         s_ek21[i, j] = area * (lamda * c[0, j] * c[1, i] + mu * c[1, j] * c[0, i])
    #         dx_e[i, j] = area / 3 * (c[0, j] * Nk[i])
    #         dy_e[i, j] = area / 3 * (c[1, j] * Nk[i])
    #
    # return s_ek11, s_ek12, s_ek21, s_ek22, dx_e, dy_e

    for i in range(3):
        # Nk[i] = c[0, i] * X[i] + c[1, i] * Y[i] + c[2, i]
        for j in range(3):
            # Nl[j] = c[0, j] * X[j] + c[1, j] * Y[j] + c[2, j]
            s_ek11[i, j] = area * ((lamda + 2 * mu) * c[0, i] * c[0, j] + mu * c[1, i] * c[1, j])
            s_ek12[i, j] = area * (lamda * c[1, i] * c[0, j] + mu * c[0, i] * c[1, j])
            s_ek22[i, j] = area * ((lamda + 2 * mu) * c[1, i] * c[1, j] + mu * c[0, i] * c[0, j])
            s_ek21[i, j] = area * (lamda * c[0, i] * c[1, j] + mu * c[1, i] * c[0, j])
            dx_e[i, j] = area / 3 * (c[0, j] * Nk[i])
            dy_e[i, j] = area / 3 * (c[1, j] * Nk[i])

    return s_ek11, s_ek12, s_ek21, s_ek22, dx_e, dy_e


def assemble_stiffness_matrix(nnodes, p, t, lamda, mu):
    s11 = np.zeros((nnodes, nnodes))
    s12 = np.zeros((nnodes, nnodes))
    s21 = np.zeros((nnodes, nnodes))
    s22 = np.zeros((nnodes, nnodes))
    dx = np.zeros((nnodes, nnodes))
    dy = np.zeros((nnodes, nnodes))

    for k in range(len(t[0])):
        el1 = p[:, t[0, k]]
        el2 = p[:, t[1, k]]
        el3 = p[:, t[2, k]]
        # el = [el1, el2, el3]
        s_ek11, s_ek12, s_ek21, s_ek22, dx_e, dy_e = generate_stiffness_matrix(el1, el2, el3, lamda, mu)

        for i in range(3):
            for j in range(3):
                s11[t[i, k], t[j, k]] = s11[t[i, k], t[j, k]] + s_ek11[i, j]
                s12[t[i, k], t[j, k]] = s12[t[i, k], t[j, k]] + s_ek12[i, j]
                s21[t[i, k], t[j, k]] = s21[t[i, k], t[j, k]] + s_ek21[i, j]
                s22[t[i, k], t[j, k]] = s22[t[i, k], t[j, k]] + s_ek22[i, j]
                dx[t[i, k], t[j, k]] = dx[t[i, k], t[j, k]] + dx_e[i, j]
                dy[t[i, k], t[j, k]] = dy[t[i, k], t[j, k]] + dy_e[i, j]

    return s11, s12, s21, s22, dx, dy


def generate_element_vector_x(el1, el2, el3, p):
    X = [el1[0], el2[0], el3[0]]
    Y = [el1[1], el2[1], el3[1]]
    area = PolyArea(X, Y)
    g = np.zeros((nnodes, 1))

    for i in range(3):
        if X[i] == 1:
            g[i] = -1

    f_ek = area / 3 * g

    return f_ek


def generate_element_vector_y(el1, el2, el3, p):
    X = [el1[0], el2[0], el3[0]]
    Y = [el1[1], el2[1], el3[1]]
    area = PolyArea(X, Y)
    g = np.zeros((nnodes, 1))

    for i in range(3):
        if Y[i] == 1:
            g[i] = -1

    f_ek = area / 3 * g

    return f_ek


def assemble_vector_x(nnodes, p, t):
    f = np.zeros((nnodes, 1))

    for k in range(len(t[0])):
        el1 = p[:, t[0, k]]
        el2 = p[:, t[1, k]]
        el3 = p[:, t[2, k]]
        f_ek = generate_element_vector_x(el1, el2, el3, p)

        for i in range(3):
            f[t[i, k]] = f[t[i, k]] + f_ek[i]

    return f


def assemble_vector_y(nnodes, p, t):
    f = np.zeros((nnodes, 1))

    for k in range(len(t[0])):
        el1 = p[:, t[0, k]]
        el2 = p[:, t[1, k]]
        el3 = p[:, t[2, k]]
        f_ek = generate_element_vector_y(el1, el2, el3, p)

        for i in range(3):
            f[t[i, k]] = f[t[i, k]] + f_ek[i]

    return f


mesh = loadmat('rounded_cave3.mat')
stif = loadmat('stif.mat')

p = mesh['p']
e = mesh['e']
t = mesh['t']
t = t - 1

S11 = stif['S11']
S12 = stif['S12']
S21 = stif['S21']
S22 = stif['S22']
fxo = stif['fxo']
fyo = stif['fyo']

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

nnodes = p.shape[1]
bn = np.array([8, 222, 9, 221, 1, 181, 11, 182, 12, 183, 13, 184, 2, 224, 10, 223, 7])  # cavern boundary nodes
bn = bn - 1

# TODO: add boundary forces

L_bnd, R_bnd, B_bnd, T_bnd = extract_bnd(p, nnodes)
s11, s12, s21, s22, dx, dy = assemble_stiffness_matrix(nnodes, p, t, lamda, mu)
fx = assemble_vector_x(nnodes, p, t)
fy = assemble_vector_y(nnodes, p, t)

# Applying BC
y_bnd = np.concatenate((B_bnd, T_bnd))

s11[L_bnd, :] = 0
s11[:, L_bnd] = 0
s11[L_bnd, L_bnd] = 1
s21[L_bnd, :] = 0
s21[:, L_bnd] = 0
s21[L_bnd, L_bnd] = 1
s12[y_bnd, :] = 0
s12[:, y_bnd] = 0
s12[y_bnd, y_bnd] = 1
s22[y_bnd, :] = 0
s22[:, y_bnd] = 0
s22[y_bnd, y_bnd] = 1

fx[L_bnd] = 0
fy[y_bnd] = 0

# dif_s11 = np.amax(s11 - S11)
# dif_s12 = np.amax(s12 - S12)
# dif_s21 = np.amax(s21 - S21)
# dif_s22 = np.amax(s22 - S22)

# Calculations
# s1 = s11 + s21
# s2 = s12 + s22
# s = np.concatenate((s1, s2), axis=1)
# f = fx + fy

# s1 = np.concatenate((s11, s12), axis=1)
# s2 = np.concatenate((s21, s22), axis=1)
# s = np.concatenate((s1, s2), axis=0)
# f = np.concatenate((fx, fy), axis=0)
# u1 = np.linalg.solve(s1, fx)
# u2 = np.linalg.solve(s2, fy)
# u = u1 + u2

s1 = [s11, s12]
s2 = [s21, s22]
s = np.column_stack((s1, s2))
f = [fx, fy]

nz11 = np.count_nonzero(s11)
nz12 = np.count_nonzero(s12)
nz21 = np.count_nonzero(s21)
nz22 = np.count_nonzero(s22)

t1 = "Number of nonzero values in s11 = {}".format(nz11)
t2 = "Number of nonzero values in s12 = {}".format(nz12)
t3 = "Number of nonzero values in s21 = {}".format(nz21)
t4 = "Number of nonzero values in s22 = {}".format(nz22)

# plt.subplot(221)
# plt.spy(s11)
# plt.title(t1)
# plt.subplot(222)
# plt.spy(s12)
# plt.title(t2)
# plt.subplot(223)
# plt.spy(s21)
# plt.title(t3)
# plt.subplot(224)
# plt.spy(s22)
# plt.title(t4)
#
# plt.show()

u = np.linalg.solve(s, f)

# s1 = [[s11], [s12]]
# s2 = [[s21], [s22]]
# A = np.ones((100, 100))
# B = np.ones((100, 1))
# C = np.linalg.solve(A, B)
# u1 = np.linalg.solve(s1, fx)
# u2 = np.linalg.solve(s2, fy)
# u = u1 + u2


print("done")
