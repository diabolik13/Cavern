import numpy as np
import sympy as sp
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import sys


# Linear elastic model functions:
def load_mesh(mesh_filename):
    """Loads mesh data: points and elements."""

    ext = mesh_filename.split(".")[-1]
    if ext.lower() == 'msh':
        m = meshio.read('./mesh/' + mesh_filename)
        p = m.points.transpose() * 1e3
        p = np.delete(p, 2, axis=0)
        t = m.cells["triangle"].transpose()
    elif ext.lower() == 'mat':
        m = loadmat('/mesh/' + mesh_filename)
        p = m['p'] * 1e3
        # e = m['e']  # edges data
        t = m['t']
        t = t - 1  # update elements numbering to start with 0
        t = np.delete(t, 3, axis=0)  # remove sub domain index (not necessary)
    else:
        sys.exit("Mesh type is not recognized.")
        # warnings.showwarning('Mesh type is not recognized')

    return m, p, t


def extract_bnd(p, dof):
    """Extracts indices of dof on the domain's boundary, such that L_bnd and R_bnd contain x-dof indices and
    B_bnd and T_bnd contain y-dof indices."""

    nnodes = p.shape[1]  # number of nodes
    l_bnd = np.array([], dtype='i')
    r_bnd = np.array([], dtype='i')
    b_bnd = np.array([], dtype='i')
    t_bnd = np.array([], dtype='i')

    # Normal case:
    for node in range(nnodes):
        if p[0][node] == min(p[0]):
            # l_bnd = np.append(l_bnd, node * dof)
            l_bnd = np.append(l_bnd, node * dof + 1)        # added rule
        if p[1][node] == min(p[1]):
            b_bnd = np.append(b_bnd, node * dof)            # added rule
            b_bnd = np.append(b_bnd, node * dof + 1)
        if p[0][node] == max(p[0]):
            r_bnd = np.append(r_bnd, node * dof)
            r_bnd = np.append(r_bnd, node * dof + 1)        # added rule
        if p[1][node] == max(p[1]):
            t_bnd = np.append(t_bnd, node * dof)            # added rule
            t_bnd = np.append(t_bnd, node * dof + 1)

    # d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))

    return l_bnd, r_bnd, b_bnd, t_bnd


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


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
    pg = np.unique(ph_group)
    pg = pg[pg < 2]     # for new cave.msh (2)
    # pg = pg[pg < 13]    # cavern(s) index is represented by physical group 11 (and 12 in case of 2 caverns)
    lines = m.cells['line']  # nodes of the boundary lines      (is node_ind)
    # new_array = np.concatenate((ph_group.reshape(len(ph_group), 1), lines), axis=1)
    nind_c1 = np.array([], dtype='i')  # cavern nodes indexes
    nind_c2 = np.array([], dtype='i')   # 2 cavern nodes indexes

    for i in range(len(lines)):
        if len(pg) == 2:
            if ph_group[i] == pg[0]:    # 1st Cavern's wall nodes group
                nind_c1 = np.append(nind_c1, lines[i, :])
            if ph_group[i] == pg[1]:    # 2nd Cavern's wall nodes group
                nind_c2 = np.append(nind_c2, lines[i, :])
        elif len(pg) == 1:  # only 1 cavern present in the mesh
            if ph_group[i] == pg[0]:
                nind_c1 = np.append(nind_c1, lines[i, :])

    nind_c1 = np.unique(nind_c1)
    nind_c2 = np.unique(nind_c2)
    nind_c = np.concatenate((nind_c1, nind_c2), axis=0)

    alpha = np.array([])
    d = np.array([])

    # copy from here:
    for i in nind_c:
        index = np.where((lines == i))[0]
        nindex = lines[index].flatten()
        seen = set([i])
        neighbours = [x for x in nindex if x not in seen and not seen.add(x)]

        if x[i] < max(x) and x[i] > min(x):
            tested = x[neighbours[0]] - x[i]
            tested1 = y[i] - y[neighbours[0]]
            tested2 = x[i] - x[neighbours[1]]
            tested3 = y[neighbours[1]] - y[i]
            if tested == 0:
                tested = 1e-5
            if tested1 == 0:
                tested1 = 1e-5
            if tested2 == 0:
                tested2 = 1e-5
            if tested3 == 0:
                tested3 = 1e-5
            # alpha1 = np.arctan((x[neighbours[0]] - x[i]) / (y[i] - y[neighbours[0]])) # o.g.
            alpha1 = np.arctan(tested / tested1)    # TODO make this look more dandy
        #     alpha2 = np.arctan((x[i] - x[neighbours[1]]) / (y[neighbours[1]] - y[i])) # o.g.
            alpha2 = np.arctan(tested2 / tested3)
            d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
            d2 = np.sqrt((x[i] - x[neighbours[1]]) ** 2 + (y[i] - y[neighbours[1]]) ** 2)
            d = np.append(d, (d1 + d2) / 2)
            if i in nind_c1:
                alpha = np.append(alpha, ((alpha1 + alpha2) / 2))
            elif i in nind_c2:
                alpha = np.append(alpha, -np.pi + ((alpha1 + alpha2) / 2))

        elif x[i] == max(x) or x[i] == min(x):
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


def elastic_moduli(Kb, mu):
    E = (9 * Kb * mu) / (3 * Kb + mu)  # Young's modulus, Pa
    v = (3 * Kb - 2 * mu) / (2 * (3 * Kb + mu))  # Poisson's ratio, -

    D = (E / (1 - v ** 2)) * np.array([[1, v, 0],
                                       [v, 1, 0],
                                       [0, 0, ((1 - v) / 2)]])      # for plane stress
    return D


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
    test = np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])
    b = 1 / np.linalg.det(j) * np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])
    return b


def generate_element_stiffness_matrix(el, d):
    b = generate_displacement_strain_matrix(el)
    x = np.array(el[:, 0])
    y = np.array(el[:, 1])
    area = polyarea(x, y)

    ke = area * np.dot((np.dot(b.transpose(), d)), b)
    return ke


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


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_singularity(a):
    return np.linalg.inv(a)


def assemble_vector(p, t, rho, g, nind_c, px=0, py=0):
    nnodes = p.shape[1]  # number of nodes
    nele = len(t[0])  # number of elements
    f = np.zeros((2 * nnodes, 1))
    pressurex = 5e6  # predefined value (o.g. 5e6) 10e6
    pressurey = pressurex  # on both sides of the rectangle similar pressure values (original case, Khaledi 10e6)
    # test = np.where(t == 4)  # find the appropriate element

    for k in range(nele):
        el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
        x = np.array(el[:, 0])
        y = np.array(el[:, 1])
        area = polyarea(x, y)
        node = np.array([t[0, k], t[1, k], t[2, k]])
        ind = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
        fe = np.zeros(6)
        j = 0
        n = 1

        for i in range(3):

            # Compression Case (rectangular mesh)
            # Applying Neumann BC to both topside and right side
            # if x[i] == np.max(p[0]) and x[i-1] == np.max(p[0]):
            #     L = np.abs(y[i] - y[i-1])
            #     fe[j] = -pressurex * (L/2)
            # if x[i] == np.max(p[0]) and x[i-2] == np.max(p[0]):
            #     L = np.abs(y[i] - y[i - 2])
            #     fe[j] = -pressurex * (L / 2)
            # if y[i] == np.max(p[1]) and y[i-2] == np.max(p[1]):
            #     L = np.abs(x[i] - x[i-2])
            #     fe[n] = -pressurey * (L/2)
            # if y[i] == np.max(p[1]) and y[i - 1] == np.max(p[1]):
            #     L = np.abs(x[i] - x[i-1])
            #     fe[n] = -pressurey * (L/2)
            # if x[i] == np.max(p[0]) and y[i] == np.max(p[1]) and x[i-2] == np.max(p[0]):  # (4)
            #     L = np.abs(y[i] - y[i-2])
            #     fe[j] = -pressurex * (L / 2)
            # if x[i] == np.max(p[0]) and y[i] == np.max(p[1]) and y[i-1] == np.max(p[1]):  # (5)
            #     L = np.abs(x[i] - x[i-1])
            #     fe[n] = -pressurey * (L/2)
            # if x[i] == np.min(p[0]) and y[i] == np.max(p[1]) and y[i-2] == np.max(p[1]):
            #     L = np.abs(x[i] - x[i-2])
            #     fe[n] = -pressurey * (L / 2)
            # if x[i] == np.max(p[0]) and y[i] == np.min(p[1]) and x[i-1] == np.max(p[0]):
            #     L = np.abs(y[i] - y[i - 1])
            #     fe[j] = -pressurex * (L / 2)
            # n = n + 2
            # j = j + 2

            # Normal Case:
            # Applying Newman's B.C. on the cavern's wall (Pressure inside the cavern, cavern mesh)
            if node[i] in nind_c:
               fe[2 * i] = px[np.where(nind_c == node[i])] # + rho * g * (y[i] - 1000)    # remove rho * g if without litho pressure
               fe[2 * i + 1] = py[np.where(nind_c == node[i])]

        for i in range(6):
            f[ind[i]] += fe[i]

    return f


def impose_dirichlet(k, f, fixed_bnd):
    """Impose Dirichlet boundary conditions."""

    k[fixed_bnd, :] = 0
    k[:, fixed_bnd] = 0
    k[fixed_bnd, fixed_bnd] = 1
    f[fixed_bnd] = 0

    return k, f


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
        strain[:, [k]] = -np.dot(b, q)
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
    lw = 0.7

    im = axs.tricontourf(triang, f, n, cmap='plasma')
    axs.triplot(triang, lw=lw)
    axs.set_title('Strain in Y')
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    plt.colorbar(im, cax=cax)
    plt.show()


def deformed_mesh(p, t, u):
    nnodes = p.shape[1]
    ux = u[::2].reshape(nnodes, )
    uy = u[1::2].reshape(nnodes, )

    # non deformed mesh coordinates
    X = p[0, :]
    Y = p[1, :]

    el0_x = np.array([X[80], X[82], X[81]])
    el0_y = np.array([Y[80], Y[82], Y[81]])
    el388_x = np.array([X[4], X[130], X[0]])
    el388_y = np.array([Y[4], Y[130], Y[0]])
    el396_x = np.array([X[354], X[41], X[3]])
    el396_y = np.array([Y[354], Y[41], Y[3]])
    el0_area = polyarea(el0_x, el0_y)
    el388_area = polyarea(el388_x, el388_y)
    el396_area = polyarea(el396_x, el396_y)

    # deformed mesh coordinates
    x = p[0, :] + ux
    y = p[1, :] + uy

    el0_x_def = np.array([x[80], x[82], x[81]])
    el0_y_def = np.array([y[80], y[82], y[81]])
    el388_x_def = np.array([x[4], x[130], x[0]])
    el388_y_def = np.array([y[4], y[130], y[0]])
    el396_x_def = np.array([x[354], x[41], x[3]])
    el396_y_def = np.array([y[354], y[41], y[3]])
    el0_area_def = polyarea(el0_x_def, el0_y_def)
    el388_area_def = polyarea(el388_x_def, el388_y_def)
    el396_area_def = polyarea(el396_x_def, el396_y_def)

    triang_1 = mtri.Triangulation(X, Y, t.transpose())
    triang = mtri.Triangulation(x, y, t.transpose())
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.triplot(triang_1, 'k-')
    axs.triplot(triang, 'r-')
    axs.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()


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


def gauss_stress_strain_tot(p, t, u, d, evp_tot):
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
        strain[:, [k]] = -np.dot(b, q)
        strain[:, [k]] = strain[:, [k]] + evp_tot[:, [k]]
        stress[:, [k]] = np.dot(d, strain[:, [k]])

    return strain, stress
