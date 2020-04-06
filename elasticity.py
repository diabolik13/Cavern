import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import sympy as sp
import meshio
import sys
import warnings

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
# from sympy import *

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D


def load_input(mesh_filename, c1, c2):
    """Load the input data."""

    # rho = 2160  # rock density, [kg/m3]
    temp = 353  # temperature, [K]
    q = 35000  # creep activation energy, [cal/mol]
    r = 1.987  # gas constant, [cal/(mol*K)]
    kb = 22e9  # Bulk modulus, [Pa]
    mu = 11e9  # Shear modulus, [Pa]
    pr = 5e6  # cavern and lithostatic pressure difference, [Pa]
    dof = 2  # degrees of freedom, [-]
    nt = 1  # number of time steps, [-]
    a = 1e-21  # creep material constant, [Pa]^n
    n = 5  # creep material constant, [-]
    th = 1e3  # thickness of the model in z, [m]
    w = 1e2  # cavern width in z, [m]
    dt = 31536000e-2  # time step, [s]
    c = 0  # wave number, frequency of loading cycles control, [-]
    cfl = 0.5  # CFL
    conv = 3e-3  # convergence threshold
    max_iter = 15  # maximum number of NR iterations

    m, p, t = load_mesh(mesh_filename)
    nnodes = p.shape[1]  # number of nodes
    x = p[0, :]
    y = p[1, :]
    lamda, e, nu, d = lame(kb, mu, plane_stress=True)

    xsym, ysym = sp.symbols('xsym ysym')
    # c1 = 0.5
    # c2 = - np.pi / 2
    # c1 = 1
    # c2 = 0
    # u_x = 1e-5 * sp.sin(np.pi * xsym / 1000) * sp.sin(np.pi * ysym / 1000)
    # u_y = 1e-5 * sp.sin(np.pi * xsym / 1000) * sp.sin(np.pi * ysym / 1000)
    u_x = 1e-5 * sp.sin(c1 * np.pi * xsym / 1000 + c2) * sp.sin(c1 * np.pi * ysym / 1000 + c2)
    u_y = 1e-5 * sp.sin(c1 * np.pi * xsym / 1000 + c2) * sp.sin(c1 * np.pi * ysym / 1000 + c2)

    # ux_anl = sp.lambdify([xsym, ysym], u_x, "numpy")(x, y)
    # uy_anl = sp.lambdify([xsym, ysym], u_y, "numpy")(x, y)
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
    f = assemble_vector(p, t, fx, fy, th, simple=False)
    l_bnd, r_bnd, b_bnd, t_bnd = extract_bnd(p, dof)
    d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))
    k = assemble_stiffness_matrix(dof, p, t, d, th)
    k, f = impose_dirichlet(k, f, d_bnd)

    input = {
        'mesh data': m,
        'time step size': dt,
        'number of time steps': nt,
        'thickness': th,
        'points': p,
        'cavern pressure': pr,
        'cavern width': w,
        'cavern temperature': temp,
        'creep activation energy': q,
        'gas constant': r,
        'elements': t,
        'material constant': a,
        'material exponent': n,
        'elasticity tensor': d,
        'shear moduli': mu,
        'Lame parameter': lamda,
        'external forces': f,
        'stiffness matrix': k,
        'Dirichlet boundaries': d_bnd,
        'CFL': cfl,
        'wave number': c1,
        'convergence threshold': conv,
        'number of iterations': max_iter,
        'ux': u_x,
        'uy': u_y,
        'strain_anl': strain_anl,
        'stress_anl': stress_anl
    }

    return input


def lame(k, mu, plane_stress):
    """Calculates lame parameters, elasticity tensor etc."""

    lamda = k - 2 / 3 * mu  # Elastic modulus
    e = mu * (3 * lamda + 2 * mu) / (lamda + mu)  # Young's modulus
    nu = lamda / (2 * (lamda + mu))  # Poisson's ratio

    if plane_stress:
        # Plain stress elasticity tensor:
        d = np.array([[lamda + 2 * mu, lamda, 0],
                      [lamda, lamda + 2 * mu, 0],
                      [0, 0, mu]])
    else:
        # Plain strain elasticity tensor:
        d = e / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0],
                                                      [nu, 1 - nu, 0],
                                                      [0, 0, (1 - 2 * nu) / 2]])
    return lamda, e, nu, d


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def extract_bnd(p, dof):
    """Extracts indices of dof on the domain's boundary, such that L_bnd and R_bnd contain x-dof indices and
    B_bnd and T_bnd contain y-dof indices."""

    nnodes = p.shape[1]  # number of nodes
    l_bnd = np.array([], dtype='i')
    r_bnd = np.array([], dtype='i')
    b_bnd = np.array([], dtype='i')
    t_bnd = np.array([], dtype='i')

    for node in range(nnodes):
        if p[0][node] == min(p[0]):
            l_bnd = np.append(l_bnd, node * dof)
            # l_bnd = np.append(l_bnd, node * dof + 1)
        if p[1][node] == min(p[1]):
            # b_bnd = np.append(b_bnd, node * dof)
            b_bnd = np.append(b_bnd, node * dof + 1)
        if p[0][node] == max(p[0]):
            r_bnd = np.append(r_bnd, node * dof)
            # r_bnd = np.append(r_bnd, node * dof + 1)
        if p[1][node] == max(p[1]):
            # t_bnd = np.append(t_bnd, node * dof)
            t_bnd = np.append(t_bnd, node * dof + 1)

    # d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))

    return l_bnd, r_bnd, b_bnd, t_bnd


def generate_element_stiffness_matrix(el, d):
    b = generate_displacement_strain_matrix(el)
    x = np.array(el[:, 0])
    y = np.array(el[:, 1])
    area = polyarea(x, y)

    ke = area * np.dot((np.dot(b.transpose(), d)), b)

    return ke


def generate_displacement_strain_matrix(el):
    x = np.array(el[:, 0])  # nodal coordinates
    y = np.array(el[:, 1])  # nodal coordinates
    xc = np.zeros((3, 3))
    yc = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            xc[i, j] = x[i] - x[j]
            yc[i, j] = y[i] - y[j]

    j = [[xc[0, 2], yc[0, 2]],
         [xc[1, 2], yc[1, 2]]]
    b = 1 / np.linalg.det(j) * np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])

    return b


def assemble_stiffness_matrix(dof, p, t, d, th):
    nnodes = p.shape[1]  # number of nodes
    nele = len(t[0])  # number of elements
    k = np.zeros((dof * nnodes, dof * nnodes))

    for e in range(nele):
        el = np.array([p[:, t[0, e]], p[:, t[1, e]], p[:, t[2, e]]])
        node = np.array([t[0, e], t[1, e], t[2, e]])
        ind = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        ke = generate_element_stiffness_matrix(el, d)

        for i in range(6):
            for j in range(6):
                k[ind[i], ind[j]] = k[ind[i], ind[j]] + ke[i, j] * th

    return k


def assemble_vector(p, t, px, py, th, simple):
    nnodes = p.shape[1]  # number of nodes
    nele = len(t[0])  # number of elements
    f = np.zeros((2 * nnodes, 1))

    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        x = np.array(el[:, 0])
        y = np.array(el[:, 1])
        area = polyarea(x, y)
        node = np.array([t[0, k], t[1, k], t[2, k]])
        ind = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        fe = np.zeros(6)
        j = 0

        for i in range(3):
            if simple:
                # Applying Newman's B.C. on the right edge
                if x[i] == 1000:
                    fe[j] = -1
                j = j + 2
            else:
                # Applying Newman's B.C. on the cavern's wall (Pressure inside the cavern)
                # if node[i] in nind_c:
                fe[2 * i] = fe[2 * i] + th * area / 3 * px[node[i]]
                fe[2 * i + 1] = fe[2 * i + 1] + th * area / 3 * py[node[i]]

        for i in range(6):
            f[ind[i]] = fe[i]
            # f[ind[i]] = f[ind[i]] + fe[i]

    return f


def gauss_stress_strain(p, t, u, d):
    """Stress and strains evaluated at Gaussian points."""

    nele = len(t[0])
    strain = np.zeros((3, nele))
    stress = np.zeros((3, nele))
    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        b = generate_displacement_strain_matrix(el)
        q = np.array([u[node[0] * 2], u[node[0] * 2 + 1],
                      u[node[1] * 2], u[node[1] * 2 + 1],
                      u[node[2] * 2], u[node[2] * 2 + 1], ])
        strain[:, [k]] = np.dot(b, q)
        stress[:, [k]] = np.dot(d, strain[:, [k]])

    return strain, stress


def nodal_stress_strain(p, t, straing, stressg):
    """Stress and strains extrapolated to nodal points."""

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
            x = np.array(el[:, 0])
            y = np.array(el[:, 1])
            area[ii] = polyarea(x, y)
            straine[:, ii] = straing[:, ind]
            stresse[:, ii] = stressg[:, ind]

        areat = np.sum(area)
        strain[:, [k]] = np.dot(straine, ((1 / areat) * area))
        stress[:, [k]] = np.dot(stresse, ((1 / areat) * area))

    return strain, stress


def plot_results(p, t, t1, t2, t3, u=0, strain=0, stress=0):
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
    n = 50
    lw = 0.2

    im = axs[0, 0].tricontourf(triang, ux, n, cmap='plasma')
    axs[0, 0].triplot(triang, lw=lw)
    axs[0, 0].set_title(t1 + ' in X')
    axs[0, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 0].tricontourf(triang, uy, n, cmap='plasma')
    axs[1, 0].triplot(triang, lw=lw)
    axs[1, 0].set_title(t1 + ' in Y')
    axs[1, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 1].tricontourf(triang, strainx, n, cmap='plasma')
    axs[0, 1].triplot(triang, lw=lw)
    axs[0, 1].set_title(t2 + ' in X')
    axs[0, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 1].tricontourf(triang, strainy, n, cmap='plasma')
    axs[1, 1].triplot(triang, lw=lw)
    axs[1, 1].set_title(t2 + ' in Y')
    axs[1, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 2].tricontourf(triang, stressx, n, cmap='plasma')
    axs[0, 2].triplot(triang, lw=lw)
    axs[0, 2].set_title(t3 + ' in X')
    axs[0, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 2].tricontourf(triang, stressy, n, cmap='plasma')
    axs[1, 2].triplot(triang, lw=lw)
    axs[1, 2].set_title(t3 + ' in Y')
    axs[1, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()


def plot_results2(p, t, t1, t2, t3, u=0, strain=0, stress=0):
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
    n = 50
    lw = 0.2

    im = axs[0, 0].tricontourf(triang, ux, n, cmap='plasma')
    axs[0, 0].triplot(triang, lw=lw)
    axs[0, 0].set_title(t1 + ' in X')
    axs[0, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 0].tricontourf(triang, uy, n, cmap='plasma')
    axs[1, 0].triplot(triang, lw=lw)
    axs[1, 0].set_title(t1 + ' in Y')
    axs[1, 0].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 1].tricontourf(triang, strainx, n, cmap='plasma')
    axs[0, 1].triplot(triang, lw=lw)
    axs[0, 1].set_title(t2 + ' in X')
    axs[0, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 1].tricontourf(triang, strainy, n, cmap='plasma')
    axs[1, 1].triplot(triang, lw=lw)
    axs[1, 1].set_title(t2 + ' in Y')
    axs[1, 1].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[0, 2].tricontourf(triang, stressx, n, cmap='plasma')
    axs[0, 2].triplot(triang, lw=lw)
    axs[0, 2].set_title(t3 + ' in X')
    axs[0, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    im = axs[1, 2].tricontourf(triang, stressy, n, cmap='plasma')
    axs[1, 2].triplot(triang, lw=lw)
    axs[1, 2].set_title(t3 + ' in Y')
    axs[1, 2].set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()


def plot_parameter(p, t, f):
    # nnodes = p.shape[1]  # number of nodes
    f = f.reshape((len(p[0]),))
    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes

    # Plot the triangulation.
    triang = mtri.Triangulation(x, y, t.transpose())

    # Set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=1)
    n = 50
    lw = 0.2

    im = axs.tricontourf(triang, f, n, cmap='plasma')
    axs.triplot(triang, lw=lw)
    axs.set_title('Plot')
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # plt.show()


def check_matrix(k):
    """Checks stiffness matrix.

    Returns:
        nz_k: number of nonzero values.
        det: determinant of k.
    """

    nz_k = np.count_nonzero(k)
    det = np.linalg.det(k)
    txt = "Number of nonzero values in k = {}, det = {}".format(nz_k, det)
    plt.spy(k)
    plt.title(txt)
    plt.show()


def load_mesh(mesh_filename):
    """Loads mesh data: points and elements."""

    ext = mesh_filename.split(".")[-1]
    if ext.lower() == 'msh':
        m = meshio.read(mesh_filename)
        p = m.points.transpose() * 1e3
        p = np.delete(p, 2, axis=0)
        t = m.cells["triangle"].transpose()
    elif ext.lower() == 'mat':
        m = loadmat(mesh_filename)
        p = m['p'] * 1e3
        # e = m['e']  # edges data
        t = m['t']
        t = t - 1  # update elements numbering to start with 0
        t = np.delete(t, 3, axis=0)  # remove sub domain index (not necessary)
    else:
        sys.exit("Mesh type is not recognized.")
        # warnings.showwarning('Mesh type is not recognized')

    return m, p, t


# def list_duplicates(seq):
#     seen = set()
#     seen_add = seen.add
#     # adds all elements it doesn't know yet to seen and all other to seen_twice
#     seen_twice = set(x for x in seq if x in seen or seen_add(x))
#     # turn the set into a list (as requested)
#     return list(seen_twice)


def cavern_boundaries(m, p, pr, w):
    """Calculate nodal forces on the domain's boundaries.

    Parameters:
        m: mesh data
        p: points data
        pr: cavern's pressure, [Pa]
        w: cavern's width, [m]ri

    Returns:
        px: x component of nodal forces
        py: y component of nodal forces
        nind_c: indexes of cavern's boundary nodes
    """

    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    ph_group = m.cell_data['line']['gmsh:physical']  # physical group of a line
    lines = m.cells['line']  # nodes of the boundary lines
    # new_array = np.concatenate((ph_group.reshape(len(ph_group), 1), lines), axis=1)
    nind_c = np.array([], dtype='i')  # cavern nodes indexes

    for i in range(len(lines)):
        if ph_group[i] == 1:
            nind_c = np.append(nind_c, lines[i, :])

    nind_c = np.unique(nind_c)
    alpha = np.array([])
    d = np.array([])

    for i in nind_c:
        index = np.where((lines == i))[0]
        nindex = lines[index].flatten()
        seen = set([i])
        neighbours = [x for x in nindex if x not in seen and not seen.add(x)]
        if x[i] > min(p[0]):
            alpha1 = np.arctan((x[neighbours[0]] - x[i]) / (y[i] - y[neighbours[0]]))
            alpha2 = np.arctan((x[i] - x[neighbours[1]]) / (y[neighbours[1]] - y[i]))
            d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
            d2 = np.sqrt((x[i] - x[neighbours[1]]) ** 2 + (y[i] - y[neighbours[1]]) ** 2)
            d = np.append(d, (d1 + d2) / 2)
            alpha = np.append(alpha, ((alpha1 + alpha2) / 2))

        elif x[i] == min(p[0]):
            if y[i] > 0:
                alpha = np.append(alpha, np.pi / 2)
                d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                d = np.append(d, d1)
            elif y[i] < 0:
                alpha = np.append(alpha, -np.pi / 2)
                d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                d = np.append(d, d1)

    # alpha_deg = np.degrees(alpha)
    # alpha_deg = np.column_stack((alpha_deg, nind_c))
    px = pr * np.cos(alpha) * d * w
    py = pr * np.sin(alpha) * d * w
    px = np.reshape(px, (len(px), 1))
    py = np.reshape(py, (len(py), 1))

    return px, py, nind_c


def impose_dirichlet(k, f, d_bnd):
    """Impose Dirichlet boundary conditions."""

    k[d_bnd, :] = 0
    k[:, d_bnd] = 0
    k[d_bnd, d_bnd] = 1
    f[d_bnd] = 0

    return k, f


def von_mises_stress(stress):
    dstress = deviatoric_stress(stress)
    stressx = stress[0, :]
    stressy = stress[1, :]
    stressxy = stress[2, :]
    # svm = np.sqrt(np.square(stressx) - stressx * stressy + np.square(stressy) + 3 * np.square(stressxy))
    svm = np.sqrt(3 / 2 * np.sum((np.transpose(dstress) * np.transpose(dstress)), axis=1))

    return svm


def deviatoric_stress(stress):
    stressx = stress[0, :]
    stressy = stress[1, :]
    stressxy = stress[2, :]
    dstressx = stressx - 0.5 * (stressx + stressy)
    dstressy = stressy - 0.5 * (stressx + stressy)
    dstressxy = stressxy
    dstress = [dstressx, dstressy, dstressxy]

    return dstress


def calculate_creep(input):
    """Models creep behavior for the given input."""
    print("Initializing explicit solver...")

    def calculate_timestep():
        dstress = deviatoric_stress(stress)
        svm = von_mises_stress(stress)
        g_cr = 3 / 2 * a * abs(np.power(svm, n - 2)) * svm * dstress * np.exp(- q / (r * temp))
        dt = cfl * 0.5 * np.max(abs(strain)) / np.max(abs(g_cr))

        return dt

    def calculate_pressure_forces(i, c):
        freq = np.sin(c * np.pi / 180 * i)
        px, py, nind_c = cavern_boundaries(m, p, pr, w)
        if freq < 0:
            sign = -1
            px = -px
            py = -py
        elif freq >= 0:
            sign = 1
            px = px
            py = py
        f = assemble_vector(p, t, nind_c, px, py)

        return f, sign

    m = input['mesh data']
    pr = input['cavern pressure']
    w = input['cavern width']
    temp = input['cavern temperature']
    q = input['creep activation energy']
    r = input['gas constant']
    p = input['points']
    t = input['elements']
    th = input['thickness']
    a = input['material constant']
    n = input['material exponent']
    d = input['elasticity tensor']
    # mu = input['shear moduli']
    # lamda = input['Lame parameter']
    # f = input['external forces']
    k = input['stiffness matrix']
    d_bnd = input['Dirichlet boundaries']
    nt = input['number of time steps']
    cfl = input['CFL']
    c = input['wave number']
    if 'time step size' in input:
        dt = input['time step size']

    et = [0]  # elapsed time of the simulation
    sign = np.zeros((1, nt))

    f, sign[0, 0] = calculate_pressure_forces(0, c)
    # Solve system of linear equations ku = f
    u = np.linalg.solve(k, f)  # nodal displacements vector
    # Postprocessing for stresses and strains evaluation
    straing, stressg = gauss_stress_strain(p, t, u, d)
    strain, stress = nodal_stress_strain(p, t, straing, stressg)

    nnodes = len(stress[0])
    nele = len(t[0])
    disp_out = np.zeros((2 * nnodes, nt))
    stress_out = np.zeros((3 * nnodes, nt))
    strain_out = np.zeros((3 * nnodes, nt))
    forces_out = np.zeros((2 * nnodes, nt))
    svm_out = np.zeros((nnodes, nt))
    # strain_cr = np.zeros((3, nnodes))
    strain_crg = np.zeros((3, nele))
    # fo = f

    # output
    disp_out[:, 0] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
    strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
    stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
    svm_out[:, 0] = von_mises_stress(stress).transpose()

    if nt > 1:
        for i in range(nt - 1):
            # calculate time step size
            if 'time step size' not in input:
                dt = calculate_timestep()

            fo, sign[0, i + 1] = calculate_pressure_forces((et[-1] + dt) / 86400, c)
            svm = von_mises_stress(stress)
            svmg = von_mises_stress(stressg)

            if sign[0, i] * sign[0, i + 1] > 0:
                dstressg = deviatoric_stress(stressg)
                g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * np.exp(- q / (r * temp))
                strain_crg = strain_crg + g_crg * dt
            else:
                strain_crg = np.zeros((3, nele))

            f_cr = assemble_creep_forces_vector(2, p, t, d, strain_crg, th)
            f = fo + f_cr  # calculate RHS = creep forces + external load
            f[d_bnd] = 0  # impose Dirichlet B.C. on forces vector
            u = np.linalg.solve(k, f)
            straing, stressg = gauss_stress_strain(p, t, u, d)
            strain, _ = nodal_stress_strain(p, t, straing, stressg)
            stressg = np.dot(d, (straing - strain_crg))
            _, stress = nodal_stress_strain(p, t, straing, stressg)
            disp_out[:, i + 1] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
            strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
            stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
            forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(nnodes, ), f_cr[1::2].reshape(nnodes, )), axis=0)
            svm_out[:, i + 1] = svm.transpose()

            # elapsed time
            et = np.append(et, et[-1] + dt)

            if np.max(abs(disp_out)) > 3:
                sys.exit("Unphysical solution on time step t = {}.".format(i))

    output = {
        'displacement': disp_out,
        'strain': strain_out,
        'stress': stress_out,
        'creep forces': forces_out,
        'Von Mises stress': svm_out,
        'elapsed time': et
    }

    print("Done.")

    return output


def diff_operator(p, t):
    nnodes = p.shape[1]  # number of nodes
    dx = np.zeros((nnodes, nnodes))
    dy = np.zeros((nnodes, nnodes))
    m = np.zeros((nnodes, nnodes))
    nele = len(t[0])

    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        node = np.array([t[0, k], t[1, k], t[2, k]])
        dxe, dye, me = generate_element_diff_operator(el)

        for i in range(3):
            for j in range(3):
                dx[node[i], node[j]] = dx[node[i], node[j]] + dxe[i, j]
                dy[node[i], node[j]] = dy[node[i], node[j]] + dye[i, j]
                m[node[i], node[j]] = m[node[i], node[j]] + me[i, j]

    return dx, dy, m


def generate_element_diff_operator(el):
    x = np.array(el[:, 0])
    y = np.array(el[:, 1])
    area = polyarea(x, y)

    m = [[x[0], y[0], 1],
         [x[1], y[1], 1],
         [x[2], y[2], 1]]

    c = np.linalg.solve(m, np.eye(3))

    dxe = np.zeros((3, 3))
    dye = np.zeros((3, 3))
    me = area / 3 * np.eye(3)

    for i in range(3):
        for j in range(3):
            dxe[i, j] = area / 3 * c[0, j]
            dye[i, j] = area / 3 * c[1, j]

    return dxe, dye, me


def calc_derivative(z, p, t):
    nnodes = len(p[0])
    dx, dy, m = diff_operator(p, t)
    dzdx = np.linalg.solve(m, np.dot(dx, z)).reshape(nnodes, )
    dzdy = np.linalg.solve(m, np.dot(dy, z)).reshape(nnodes, )

    return dzdx, dzdy


def dfunc(p):
    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    ux = 1e3 * np.sin(np.pi * x) * np.sin(np.pi * y)
    uy = 1e3 * np.sin(np.pi * x) * np.sin(np.pi * y)
    duxdx = 1e3 * np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    duydy = 1e3 * np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    return ux, uy, duxdx, duydy


# def creep_forces(strain_cr, p, t, mu, lamda):
#     """Calculates creep forces as tABDE."""
#
#     nnodes = len(p[0])
#     xdx, xdy = calc_derivative(strain_cr[0], p, t)
#     ydx, ydy = calc_derivative(strain_cr[1], p, t)
#     sdx, sdy = calc_derivative(strain_cr[2], p, t)
#     fcrx = ((2 * mu + lamda) * xdx + lamda * ydx + mu * sdy).reshape(nnodes, 1)
#     fcry = ((2 * mu + lamda) * ydy + lamda * xdy + mu * sdx).reshape(nnodes, 1)
#     f_cr = np.empty((fcrx.size + fcry.size, 1), dtype=fcrx.dtype)
#     f_cr[0::2] = fcrx
#     f_cr[1::2] = fcry
#
#     return f_cr


def assemble_creep_forces_vector(dof, p, t, d, ecr, th):
    nnodes = p.shape[1]  # number of nodes
    nele = len(t[0])  # number of elements
    fcr = np.zeros((dof * nnodes, 1))

    for e in range(nele):
        el = np.array([p[:, t[0, e]], p[:, t[1, e]], p[:, t[2, e]]])
        x = np.array(el[:, 0])
        y = np.array(el[:, 1])
        area = polyarea(x, y)
        global_node = np.array([t[0, e], t[1, e], t[2, e]])
        ind = [global_node[0] * 2, global_node[0] * 2 + 1, global_node[1] * 2, global_node[1] * 2 + 1,
               global_node[2] * 2, global_node[2] * 2 + 1]
        b = generate_displacement_strain_matrix(el)
        fcre = 0.5 * th * area * np.dot(np.transpose(b), np.dot(d, ecr[:, e]))

        for i in range(6):
            fcr[ind[i]] = fcr[ind[i]] + fcre[i]

    return fcr


def calculate_creep_NR(input):
    """Models creep behavior for the given input."""

    print("Initializing implicit+NR solver...")

    def calculate_timestep():
        dstress = deviatoric_stress(stress)
        svm = von_mises_stress(stress)
        g_cr = 3 / 2 * a * abs(np.power(svm, n - 2)) * svm * dstress * np.exp(- q / (r * temp))
        dt = cfl * 0.5 * np.max(abs(strain)) / np.max(abs(g_cr))

        return dt

    def calculate_pressure_forces(i, c):
        freq = np.sin(c * np.pi / 180 * i)
        px, py, nind_c = cavern_boundaries(m, p, pr, w)
        if freq < 0:
            sign = -1
            px = -px
            py = -py
        elif freq >= 0:
            sign = 1
            px = px
            py = py
        f = assemble_vector(p, t, nind_c, px, py)

        return f, sign

    m = input['mesh data']
    pr = input['cavern pressure']
    w = input['cavern width']
    temp = input['cavern temperature']
    q = input['creep activation energy']
    r = input['gas constant']
    p = input['points']
    t = input['elements']
    th = input['thickness']
    a = input['material constant']
    n = input['material exponent']
    d = input['elasticity tensor']
    # mu = input['shear moduli']
    # lamda = input['Lame parameter']
    f = input['external forces']
    k = input['stiffness matrix']
    d_bnd = input['Dirichlet boundaries']
    nt = input['number of time steps']
    cfl = input['CFL']
    # c = input['wave number']
    conv = input['convergence threshold']
    max_iter = input['number of iterations']
    if 'time step size' in input:
        dt = input['time step size']

    nnodes = len(p[0])
    nele = len(t[0])
    disp_out = np.zeros((2 * nnodes, nt))
    stress_out = np.zeros((3 * nnodes, nt))
    strain_out = np.zeros((3 * nnodes, nt))
    forces_out = np.zeros((2 * nnodes, nt))
    svm_out = np.zeros((nnodes, nt))
    strain_crg_n = np.zeros((3, nele))

    # f, _ = calculate_pressure_forces(0, 0)
    # Solve system of linear equations ku = f
    u = np.linalg.solve(k, f)  # nodal displacements vector
    # Postprocessing for stresses and strains evaluation
    straing, stressg = gauss_stress_strain(p, t, u, d)
    strain, stress = nodal_stress_strain(p, t, straing, stressg)
    fo = f
    et = [0]

    # output
    disp_out[:, 0] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
    strain_out[:, 0] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
    stress_out[:, 0] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
    svm_out[:, 0] = von_mises_stress(stress).transpose()

    if nt > 1:
        for i in range(nt - 1):
            print('Time step {}, dt = {} s:'.format(i + 1, dt))
            converged = 0
            iter = 0

            # calculate time step size
            if 'time step size' not in input:
                dt = calculate_timestep()

            while converged == 0:
                dstressg = deviatoric_stress(stressg)
                svmg = von_mises_stress(stressg)
                g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * np.exp(- q / (r * temp))
                strain_crg = strain_crg_n + g_crg * dt
                f_cr = assemble_creep_forces_vector(2, p, t, d, strain_crg, th)
                f = fo + f_cr  # calculate RHS = creep forces + external load
                f[d_bnd] = 0  # impose Dirichlet B.C. on forces vector

                residual = np.dot(k, u) - f
                delta_u = - np.linalg.solve(k, residual)

                u = u + delta_u

                # update properties
                straing, stressg = gauss_stress_strain(p, t, u, d)
                strain, _ = nodal_stress_strain(p, t, straing, stressg)
                stressg = np.dot(d, (straing - strain_crg))
                _, stress = nodal_stress_strain(p, t, straing, stressg)
                svm = von_mises_stress(stress)
                svmg = von_mises_stress(stressg)
                g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * np.exp(- q / (r * temp))
                strain_crg = strain_crg_n + g_crg * dt
                f_cr = assemble_creep_forces_vector(2, p, t, d, strain_crg, th)
                f = fo + f_cr  # calculate RHS = creep forces + external load
                f[d_bnd] = 0  # impose Dirichlet B.C. on forces vector

                # re-compute residual
                residual = np.dot(k, u) - f
                res = np.linalg.norm(residual)
                iter += 1

                print("Iteration {}, norm(residual) = {}.".format(iter, res))
                if iter == max_iter and res >= conv:
                    print("Maximum iterations reached.")
                if res < conv or iter >= max_iter:
                    converged = 1

            strain_crg_n = strain_crg
            disp_out[:, i + 1] = np.concatenate((u[::2].reshape(nnodes, ), u[1::2].reshape(nnodes, )), axis=0)
            strain_out[:, i + 1] = np.concatenate((strain[0], strain[1], strain[2]), axis=0)
            stress_out[:, i + 1] = np.concatenate((stress[0], stress[1], stress[2]), axis=0)
            forces_out[:, i + 1] = np.concatenate((f_cr[0::2].reshape(nnodes, ), f_cr[1::2].reshape(nnodes, )), axis=0)
            svm_out[:, i + 1] = svm.transpose()
            et = np.append(et, et[-1] + dt)

    output = {
        'displacement': disp_out,
        'strain': strain_out,
        'stress': stress_out,
        'creep forces': forces_out,
        'Von Mises stress': svm_out,
        'elapsed time': et
    }

    print("Done.")

    return output


def plot_results(u, u_anl, strain, strain_anl, stress, stress_anl, nnodes, x, y, t, i):
    fig, ax = plt.subplots(nrows=4, ncols=3)

    z = u[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 0].set_title('Num Displacement X')

    z = u[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 0].set_title('Num Displacement Y')

    z = strain[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 1].set_title('Num Strain X')

    z = strain[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 1].set_title('Num Strain Y')

    z = stress[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 2].set_title('Num Stress X')

    z = stress[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 2].set_title('Num Stress Y')

    z = u_anl[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[2, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[2, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[2, 0].set_title('Anl Displacement X')

    z = u_anl[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[3, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[3, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[3, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[3, 0].set_title('Anl Displacement Y')

    z = strain_anl[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[2, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[2, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[2, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[2, 1].set_title('Anl Strain X')

    z = strain_anl[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[3, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[3, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[3, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[3, 1].set_title('Anl Strain Y')

    z = stress_anl[:nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[2, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[2, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[2, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[2, 2].set_title('Anl Stress X')

    z = stress_anl[nnodes:2 * nnodes].reshape((nnodes,))
    divider = make_axes_locatable(ax[3, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[3, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[3, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[3, 2].set_title('Anl Stress Y')

    fig.tight_layout(pad=0.1)
    plt.savefig('./results/solution_results_' + str(i) + '.png')

    # plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=3)

    z = u[:nnodes].reshape((nnodes,)) - u_anl[:nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[0, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 0].set_title('Diff Displacement X')

    z = u[nnodes:2 * nnodes].reshape((nnodes,)) - u_anl[nnodes:2 * nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 0].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 0].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 0].set_title('Diff Displacement Y')

    z = strain[:nnodes].reshape((nnodes,)) - strain_anl[:nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[0, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 1].set_title('Diff Strain X')

    z = strain[nnodes:2 * nnodes].reshape((nnodes,)) - strain_anl[nnodes:2 * nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 1].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 1].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 1].set_title('Diff Strain Y')

    z = stress[:nnodes].reshape((nnodes,)) - stress_anl[:nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[0, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[0, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[0, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[0, 2].set_title('Diff Stress X')

    z = stress[nnodes:2 * nnodes].reshape((nnodes,)) - stress_anl[nnodes:2 * nnodes].reshape((nnodes,))
    z = abs(z)
    divider = make_axes_locatable(ax[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax[1, 2].set_aspect('equal', 'box')
    triang = mtri.Triangulation(x, y, t.transpose())
    c = ax[1, 2].tricontourf(triang, z, 10, cmap='plasma', vmin=np.min(z), vmax=np.max(z),
                             levels=np.linspace(np.min(z), np.max(z), 30))
    cbar = plt.colorbar(c, cax=cax, format='%.0e', ticks=np.linspace(np.min(z), np.max(z), 3))
    ax[1, 2].set_title('Diff Stress Y')

    fig.tight_layout(pad=0.1)
    plt.savefig('./results/solution_difference_' + str(i) + '.png')
    plt.show()
    plt.close('all')


def plot_difference(size, diff_u, diff_strain, diff_stress, order_disp, order_strain, order_stress):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs = axs.flatten()

    axs[0].plot(size, diff_u, 'ro-', lw=2)
    axs[0].set_title('Consistency of displacement solution')
    axs[0].grid(True, which="both")
    axs[0].set_adjustable("datalim")
    axs[0].set_title("slope = {0:.2f}".format(order_disp))
    axs[0].set_xlabel("Element Size", fontsize=14)
    axs[0].set_ylabel("Displacement Error", fontsize=14)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

    axs[1].plot(size, diff_strain, 'ro-', lw=2)
    axs[1].set_title('Consistency of strain solution')
    axs[1].grid(True, which="both")
    axs[1].set_adjustable("datalim")
    axs[1].set_title("slope = {0:.2f}".format(order_strain))
    axs[1].set_xlabel("Element Size", fontsize=14)
    axs[1].set_ylabel("Strain Error", fontsize=14)
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')

    axs[2].plot(size, diff_stress, 'ro-', lw=2)
    axs[2].set_title('Consistency of stress solution')
    axs[2].grid(True, which="both")
    axs[2].set_title("slope = {0:.2f}".format(order_stress))
    axs[2].set_adjustable("datalim")
    axs[2].set_xlabel("Element Size", fontsize=14)
    axs[2].set_ylabel("Stress Error", fontsize=14)
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')

    plt.setp(axs, xticks=[10, 100], xticklabels=['10', '100'])
    fig.tight_layout(pad=1)
    # fig.tight_layout()
    plt.savefig('./results/approx_order.png')
    plt.show()
