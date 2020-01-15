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


def generate_stiffness_matrix(el, D):
    X = np.array(el[:, 0])
    Y = np.array(el[:, 1])
    area = PolyArea(X, Y)
    # area = 1 / 2 * abs(np.linalg.det(J))
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

    ke = area * np.dot((np.dot(B.transpose(), D)), B)

    return ke, B


def assemble_stiffness_matrix(dof, nnodes, p, t, D):
    K = np.zeros((dof * nnodes, dof * nnodes))
    nele = len(t[0])

    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        I = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        ke, B = generate_stiffness_matrix(el, D)

        for i in range(6):
            for j in range(6):
                K[I[i], I[j]] = K[I[i], I[j]] + ke[i, j]

    return K


def assemble_vector(nnodes, p, t):
    f = np.zeros((2 * nnodes, 1))
    nele = len(t[0])

    for k in range(nele):
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


def stress_strain(u, D):
    nele = len(t[0])
    strain = np.zeros((3, nele))
    stress = np.zeros((3, nele))
    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        ke, B = generate_stiffness_matrix(el, D)
        q = np.array([u[node[0] * 2], u[node[0] * 2 + 1],
                      u[node[1] * 2], u[node[1] * 2 + 1],
                      u[node[2] * 2], u[node[2] * 2 + 1], ])
        strain[:, [k]] = np.dot(B, q)
        stress[:, [k]] = np.dot(D, strain[:, [k]])

    return strain, stress


def nodal_stress_strain(p, t, straing, stressg):
    nnodes = p.shape[1]
    strain = np.zeros((3, nnodes))
    stress = np.zeros((3, nnodes))

    for k in range(nnodes):
        i, j = np.where(t == k)
        ntri = i.shape[0]
        area = np.zeros((ntri, 1))
        straine = np.zeros((3, ntri))
        stresse = np.zeros((3, ntri))

        for ii in range(ntri):
            ind = j[ii]
            el = np.array([p[:, t[0, ind]], p[:, t[1, ind]], p[:, t[2, ind]]])
            X = np.array(el[:, 0])
            Y = np.array(el[:, 1])
            area[ii] = PolyArea(X, Y)
            straine[:, ii] = straing[:, ind]
            stresse[:, ii] = stressg[:, ind]

        areat = np.sum(area)
        strain[:, [k]] = np.dot(straine, ((1 / areat) * area))
        stress[:, [k]] = np.dot(stresse, ((1 / areat) * area))

    return strain, stress


def plot_results(XX, YY):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # ===============
    #  First subplot
    # ===============
    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plot a 3D surface
    surf = ax.plot_trisurf(x, y, XX, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-100.01, 100.01)
    ax.view_init(elev=90, azim=-90)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # ===============
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # plot a 3D surface
    surf = ax.plot_trisurf(x, y, YY, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    ax.set_zlim(-100.01, 100.01)
    ax.view_init(elev=90, azim=-90)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()


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
D = np.array([[lamda + 2 * mu, lamda, 0],
              [lamda, lamda + 2 * mu, 0],
              [0, 0, mu]])

mesh = loadmat('rounded_cave3.mat')
# mesh = loadmat('R1.mat')

p = mesh['p']
e = mesh['e']
t = mesh['t']
t = t - 1  # update elements numbering to start with 0
t = np.delete(t, 3, axis=0)
dof = 2
nnodes = p.shape[1]
nele = len(t[0])
x = p[0, :]
y = p[1, :]

L_bnd, R_bnd, B_bnd, T_bnd = extract_bnd(p, nnodes, dof)
I_bnd = np.concatenate((L_bnd, R_bnd, B_bnd, T_bnd))
k = assemble_stiffness_matrix(dof, nnodes, p, t, D)
f = assemble_vector(nnodes, p, t)

# Impose Dirichlet B.C.
bnd = np.concatenate((B_bnd, T_bnd, L_bnd))
k[bnd, :] = 0
k[:, bnd] = 0
k[bnd, bnd] = 1
# f[bnd] = 0

u = np.linalg.solve(k, f)  # nodal displacements vector
straing, stressg = stress_strain(u, D)  # stress and strains evaluated at Gaussian points
strain, stress = nodal_stress_strain(p, t, straing, stressg)
strainx = strain[0, :]
strainy = strain[1, :]
stressx = stress[0, :]
stressy = stress[1, :]

ux = u[::2].reshape(nnodes, )
uy = u[1::2].reshape(nnodes, )

# displacement_plot = plot_results(ux, uy)
# strain_plot = plot_results(strainx, strainy)
# stress_plot = plot_results(stressx, stressy)

# # set up a figure twice as wide as it is tall
# fig = plt.figure(figsize=plt.figaspect(0.5))
#
# # ===============
# #  First subplot
# # ===============
# # set up the axes for the first plot
# ax = fig.add_subplot(1, 2, 1, projection='3d')
#
# # plot a 3D surface
# surf = ax.plot_trisurf(x, y, ux, cmap=cm.jet,
#                        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
# ax.view_init(elev=90, azim=-90)
# fig.colorbar(surf, shrink=0.5, aspect=10)
#
# # ===============
# # Second subplot
# # ===============
# # set up the axes for the second plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
#
# # plot a 3D surface
# surf = ax.plot_trisurf(x, y, uy, cmap=cm.jet,
#                        linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
# ax.view_init(elev=90, azim=-90)
# fig.colorbar(surf, shrink=0.5, aspect=10)
#
# plt.show()

triangles = np.zeros((nele, 3))
for i in range(nele):
    triangles[[i], :] = [t[:, i]]

triang = mtri.Triangulation(x, y, triangles)

# Set up the figure
fig, axs = plt.subplots(nrows=2, ncols=3)
axs = axs.flatten()

# Plot the triangulation.
plot1 = axs[0].tricontourf(triang, ux, 100)
axs[0].triplot(triang, lw=0.5)
axs[0].set_title('Displacement in X')
fig.tight_layout()
fig.colorbar(plot1, ax=axs[0])

plot2 = axs[3].tricontourf(triang, uy, 100)
axs[3].triplot(triang, lw=0.5)
axs[3].set_title('Displacement in Y')
fig.tight_layout()
fig.colorbar(plot2, ax=axs[3])

plot3 = axs[1].tricontourf(triang, strainx, 100)
axs[1].triplot(triang, lw=0.5)
axs[1].set_title('Strain in X')
fig.tight_layout()
fig.colorbar(plot3, ax=axs[1])

plot4 = axs[4].tricontourf(triang, strainy, 100)
axs[4].triplot(triang, lw=0.5)
axs[4].set_title('Strain in Y')
fig.tight_layout()
fig.colorbar(plot4, ax=axs[4])

plot5 = axs[2].tricontourf(triang, stressx, 100)
axs[2].triplot(triang, lw=0.5)
axs[2].set_title('Stress in X')
fig.tight_layout()
fig.colorbar(plot5, ax=axs[2])

plot6 = axs[5].tricontourf(triang, stressy, 100)
axs[5].triplot(triang, lw=0.5)
axs[5].set_title('Stress in Y')
fig.tight_layout()
fig.colorbar(plot6, ax=axs[5])
plt.show()



print("done")
