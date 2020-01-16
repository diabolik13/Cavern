import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import sys
import warnings

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lame(k, mu):
    lamda = k - 2 / 3 * mu  # Elastic modulus
    e = mu * (3 * lamda + 2 * mu) / (lamda + mu)  # Young's modulus
    nu = lamda / (2 * (lamda + mu))  # Poisson's ratio
    d = np.array([[lamda + 2 * mu, lamda, 0],
                  [lamda, lamda + 2 * mu, 0],
                  [0, 0, mu]])  # Elasticity tensor
    return lamda, e, nu, d


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def extract_bnd(p, dof):
    nnodes = p.shape[1]  # number of nodes
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


def assemble_stiffness_matrix(dof, p, t, D):
    nnodes = p.shape[1]  # number of nodes
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


def assemble_vector(p, t):
    nnodes = p.shape[1]  # number of nodes
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
            # Applying Newman's B.C. on the right edge
            if X[i] == 1:
                fe[j] = -1
            j = j + 2

        for i in range(6):
            f[I[i]] = f[I[i]] + fe[i]

    return f


def gauss_stress_strain(p, t, u, D):
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


def plot_results(p, t, u, strain, stress):
    nnodes = p.shape[1]  # number of nodes
    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes

    ux = u[::2].reshape(nnodes, )
    uy = u[1::2].reshape(nnodes, )
    strainx = strain[0, :]
    strainy = strain[1, :]
    stressx = stress[0, :]
    stressy = stress[1, :]

    # Plot the triangulation.
    triang = mtri.Triangulation(x, y, t.transpose())

    # Set up the figure
    fig, axs = plt.subplots(nrows=2, ncols=3)
    # axs = axs.flatten()
    N = 50

    im = axs[0, 0].tricontourf(triang, ux, N)
    axs[0, 0].triplot(triang, lw=0.5)
    axs[0, 0].set_title('Displacement in X')
    axs[0, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 0].tricontourf(triang, uy, N)
    axs[1, 0].triplot(triang, lw=0.5)
    axs[1, 0].set_title('Displacement in Y')
    axs[1, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 1].tricontourf(triang, strainx, N)
    axs[0, 1].triplot(triang, lw=0.5)
    axs[0, 1].set_title('Strain in X')
    axs[0, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 1].tricontourf(triang, strainy, N)
    axs[1, 1].triplot(triang, lw=0.5)
    axs[1, 1].set_title('Strain in Y')
    axs[1, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 2].tricontourf(triang, stressx, N)
    axs[0, 2].triplot(triang, lw=0.5)
    axs[0, 2].set_title('Stress in X')
    axs[0, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 2].tricontourf(triang, stressy, N)
    axs[1, 2].triplot(triang, lw=0.5)
    axs[1, 2].set_title('Stress in X')
    axs[1, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()


def check_matrix(k):
    nz_k = np.count_nonzero(k)
    det = np.linalg.det(k)
    txt = "Number of nonzero values in k = {}, det = {}".format(nz_k, det)
    plt.spy(k)
    plt.title(txt)
    plt.show()


def load_mesh(mesh_filename):
    ext = mesh_filename.split(".")[-1]
    if ext.lower() == 'msh':
        m = meshio.read(mesh_filename)
        p = m.points.transpose()
        t = m.cells["triangle"].transpose()
    elif ext.lower() == 'mat':
        m = loadmat(mesh_filename)
        p = m['p']
        # e = m['e']  # edges data
        t = m['t']
        t = t - 1  # update elements numbering to start with 0
        t = np.delete(t, 3, axis=0)  # remove sub domain index (not necessary)
    else:
        warnings.showwarning('Mesh type is not recognized')
        sys.exit()

    return m, p, t


def gmsh_boundaries(m):
    ph_group = m.cell_data['line']['gmsh:physical']  # physical group of a line
    lind = m.cells['line']  # nodes of the boundary lines
    nind_c = np.array([])  # cavern nodes indexes
    for i in range(len(lind)):
        if ph_group[i] == 1:
            nind_c = np.append(nind_c, lind[i, :])

    nind_c = np.unique(nind_c)
    return nind_c
