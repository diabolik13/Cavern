import meshio
import numpy as np
import sympy as sp

from elasticity import von_mises_stress


def polyarea(coord):
    x, y = coord
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def el_tenzor(mu, k):
    # Elastic modulus
    lamda = k - 2 / 3 * mu
    # Elasticity tenzor
    d = np.array([[lamda + 2 * mu, lamda, 0],
                  [lamda, lamda + 2 * mu, 0],
                  [0, 0, mu]])
    return d


class Mesh(object):
    """
    A mesh is a list of point global coordinates
    and a list of element definitions (by cornerpoint)
    This class loads a uniform mesh of a domain Omega
    Mesh('mesh_filename.msh')
    coordinates(n=None): return coord node n, or all coords
    cells(n=None): array of n-th cell's cornerpoint numbers, or all
    size(): returns number of elements
    """

    def __init__(self, filename, xfactor=1, yfactor=1):
        """
        N is the number of elements = number of INTERVALS
        a and b are interval endpoints
        """
        m = meshio.read(filename)

        #  Coordinates
        m.points[:, 0] *= xfactor
        m.points[:, 1] *= yfactor
        self.__nodes = m.points
        self.__nodes = np.delete(self.__nodes, 2, axis=1)
        self.__nodes = self.__nodes.transpose()

        #  Elements
        self.__cells = m.cells["triangle"].transpose()
        self.__N = len(self.__cells[0])

        #  Nodes and dofs
        self.__Nnodes = len(m.points)
        self.__Ndofs = self.__Nnodes * 2

        #  Physical groups
        self.__group = m.cell_data['line']['gmsh:physical']
        self.__pg = m.cell_data["triangle"]["gmsh:physical"]

        #  Edges
        self.__edges = m.cells['line']

        # lines and cells
        self.__cellsdata = m.cells
        self.__pointsdata = m.points

        # mesh
        self.__mesh = m

    def meshdata(self):
        return self.__mesh

    def cellsdata(self):
        return self.__cellsdata

    def pointsdata(self):
        return self.__pointsdata

    def nele(self):
        return self.__N

    def ndofs(self):
        return self.__Ndofs

    def nnodes(self):
        return self.__Nnodes

    def coordinates(self, n=None):

        if not (n is None):
            return self.__nodes[:, n]
        else:
            return self.__nodes

    def cells(self, n=None):

        if not (n is None):
            return self.__cells[:, n]
        else:
            return self.__cells

    def group(self, n=None):

        if not (n is None):
            return self.__group[n]
        else:
            return self.__group

    def edges(self, n=None):

        if not (n is None):
            return self.__edges[n]
        else:
            return self.__edges

    def line_ph_group(self, n=None):

        if not (n is None):
            return self.__group[n]
        else:
            return self.__group

    def cell_ph_group(self, n=None):

        if not (n is None):
            return self.__pg[n]
        else:
            return self.__pg

    def extract_bnd(self, lx=None, ly=None, rx=None, ry=None, tx=None, ty=None, bx=None, by=None):
        """
        Extracts indices of dof on the domain's boundary, such that L_bnd and R_bnd contain x-dof indices and
        B_bnd and T_bnd contain y-dof indices.
        """

        l_bnd_x = np.array([], dtype='i')
        l_bnd_y = np.array([], dtype='i')
        r_bnd_x = np.array([], dtype='i')
        r_bnd_y = np.array([], dtype='i')
        b_bnd_x = np.array([], dtype='i')
        b_bnd_y = np.array([], dtype='i')
        t_bnd_x = np.array([], dtype='i')
        t_bnd_y = np.array([], dtype='i')
        d_bnd = np.array([], dtype='i')
        x = self.__nodes[0]
        y = self.__nodes[1]

        for i in range(self.__Nnodes):
            if x[i] == np.min(x):
                l_bnd_x = np.append(l_bnd_x, i * 2)
                l_bnd_y = np.append(l_bnd_y, i * 2 + 1)
            if y[i] == np.min(y):
                b_bnd_x = np.append(b_bnd_x, i * 2)
                b_bnd_y = np.append(b_bnd_y, i * 2 + 1)
            if x[i] == np.max(x):
                r_bnd_x = np.append(r_bnd_x, i * 2)
                r_bnd_y = np.append(r_bnd_y, i * 2 + 1)
            if y[i] == np.max(y):
                t_bnd_x = np.append(t_bnd_x, i * 2)
                t_bnd_y = np.append(t_bnd_y, i * 2 + 1)

        # d_bnd = np.concatenate((b_bnd, t_bnd, l_bnd, r_bnd))
        if not (lx == False):
            d_bnd = np.concatenate((d_bnd, l_bnd_x))
        if not (ly == False):
            d_bnd = np.concatenate((d_bnd, l_bnd_y))
        if not (rx == False):
            d_bnd = np.concatenate((d_bnd, r_bnd_x))
        if not (ry == False):
            d_bnd = np.concatenate((d_bnd, r_bnd_y))
        if not (bx == False):
            d_bnd = np.concatenate((d_bnd, b_bnd_x))
        if not (by == False):
            d_bnd = np.concatenate((d_bnd, b_bnd_y))
        if not (tx == False):
            d_bnd = np.concatenate((d_bnd, t_bnd_x))
        if not (ty == False):
            d_bnd = np.concatenate((d_bnd, t_bnd_y))

        return d_bnd


class Shapefns(object):
    """
    Define shape functions
    These will be defined on the local coordinates
    Shapefns()
    eval(n,xi): phi[n](xi, tau)
    ddxi(n):  dphi[n]
    ddtau(n):  dphi[n]
    size(): number of nodes for these shape functions
    """

    def __init__(self):
        """
        an array of functions for phi and deriv phi
        """
        # linear shape functions
        self.__phi = [lambda xi, tau: xi,
                      lambda xi, tau: tau,
                      lambda xi, tau: 1 - xi - tau]
        # and derivatives of phi w.r.t. xi and tau
        self.__dphidxi = [1, 0, -1]
        self.__dphidtau = [0, 1, -1]
        self.__N = 3  # number of nodes in element

    def eval(self, n, xi, tau):
        """
        the function phi[n](xi, tau), for any xi and tau
        """
        return self.__phi[n](xi, tau)

    def ddxi(self, n):
        """
        the function dphidxi[n]
        """
        return self.__dphidxi[n]

    def ddtau(self, n):
        """
        the function dphidtau[n]
        """
        return self.__dphidtau[n]

    def size(self):
        """
        the  number of points
        """
        return self.__N


class FiniteElement(object):
    """
    A single finite element
    FiniteElement(mesh,sfns,eltno,dofnos): constructor
        mesh is a Mesh
        sfns is a set of shape functions
        eltno=this element number, needs to be in mesh
        dofnos=numbers of this element's dofs (numDofs-sized array)
    endpts(): cell end points
    dofnos(): all dof values
    numDofs(): number of dofs
    eval(n,x): phi[n](x)  (x, not xi)
    ddx(n,x):  dphi[n](x) (x, not xi)
    integral(f1=None,f2=None,derivative=False): integral(f1*f2*phi)
      f1, f2: ndof-sized vector of coeffs for local function
      derivative=True, do integral(f1*f2*dphi)
    """

    def __init__(self, mesh, sfns, eltno, mu, kb):
        """
        mesh is the mesh it is built on
        sfns is the Shapefuns member
        eltno is this element's number
        endnos is a pair of ints giving the numbers of the endpoints
            in the mesh
        dofnos is an array of ints giving the numbers of the dofs
        """
        assert (0 <= eltno < mesh.nele())
        # element number
        self.__eltno = eltno
        # node numbers of the element
        self.__endnos = mesh.cells(eltno)
        # check that triangular elements are used (3 nodes)
        assert (len(self.__endnos) == 3)
        # nodal x, y coordinates
        self.__endpts = np.array(mesh.coordinates(self.__endnos))
        # number of degree of freedom (6)
        self.__numDofs = 2 * sfns.size()
        # degrees of freedom indexes of the element
        self.__dofnos = np.array(
            [2 * self.__endnos[0], 2 * self.__endnos[0] + 1,
             2 * self.__endnos[1], 2 * self.__endnos[1] + 1,
             2 * self.__endnos[2], 2 * self.__endnos[2] + 1])
        # shape functions
        self.__sfns = sfns
        # area of the element
        self.__area = polyarea(mesh.coordinates(self.__endnos))
        # elasticity tenzor of the element
        self.__D = el_tenzor(mu[mesh.cell_ph_group(eltno) - 1], kb[mesh.cell_ph_group(eltno) - 1])
        # self.__D = el_tenzor(mu[0], kb[0])

    def eltno(self):
        """ access element number """
        return self.__eltno

    def el_tenz(self):
        """ access elasticity tenzor """
        return self.__D

    def endpts(self):
        """ access endpoints """
        return self.__endpts

    def nodes(self):
        """ access node indexes """
        return self.__endnos

    def area(self):
        """ evaluate area """
        return self.__area

    def dofnos(self):
        """ access dof points indexes """
        return self.__dofnos

    def numDofs(self):
        """ access numDofs """
        return self.__numDofs

    def jacobi(self, inv=False):
        """calculate J to perform local to global coordinates transformation"""
        x, y = self.__endpts
        xc = np.zeros((3, 3))
        yc = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                xc[i, j] = x[i] - x[j]
                yc[i, j] = y[i] - y[j]

        j = [[xc[0, 2], yc[0, 2]],
             [xc[1, 2], yc[1, 2]]]

        if inv:
            return np.linalg.inv(j)
        else:
            return j

    def derivative(self, n):
        """
        evaluate the n-th shape function on this element
        at the spatial coordinate x
        """

        return np.array([self.__sfns.ddxi(n), self.__sfns.ddtau(n)])


class FunctionSpace(object):
    """
    A FunctionSpace has a list of elements
    numbered and with coords according to mesh
    FunctionSpace(mesh,sfns): constructor, sfns is ShapeFuns
    size(): number of elements
    ndofs(): number of all dofs
    int_phi_phi(c=None,derivative=[False,False]):
        integral(c*phi*phi) or
        integral(c*dphi*phi) or
        integral(c*dphi*dphi) or
        integral(c*phi*dphi)
    int_phi(f=None,derivative=False):
        integral(f*phi) or
        integral(f*dphi)

    """

    def __init__(self, mesh, sfns, mu, kb):
        """
        mesh is the mesh
        sfns is the Shapefuns
        """
        # number of nodes
        self.__nnodes = mesh.nnodes()
        # number of elements
        self.__nele = mesh.nele()
        # number of degrees of freedom
        self.__nDOFs = mesh.ndofs()
        # list of all elements
        self.__elts = list([])
        # mesh
        self.__mesh = mesh
        for n in range(self.__nele):
            fe = FiniteElement(mesh, sfns, n, mu, kb)
            self.__elts.append(fe)

    def size(self):
        return len(self.__elts)

    def Ndofs(self):
        return self.__nDOFs

    def strain_disp_matrix(self, eltno):
        B = np.zeros((3, 2))
        for i in range(3):
            dNl = self.__elts[eltno].derivative(i)
            dN = np.dot(self.__elts[eltno].jacobi(inv=True), dNl)
            Bi = np.array([[dN[0], 0],
                           [0, dN[1]],
                           [dN[1], dN[0]]])
            B = np.append(B, Bi, axis=1)
        B = np.delete(B, [0, 1], axis=1)
        return B

    def stiff_matrix(self, th):
        """
        assemble stiffness matrix
        :return:
        """
        nDofs = self.__nDOFs
        k = np.zeros((nDofs, nDofs))
        for elt in self.__elts:
            D = elt.el_tenz()
            B = self.strain_disp_matrix(elt.eltno())
            ind = elt.dofnos()
            area = elt.area()
            ke = th * area * np.dot(B.transpose(), (np.dot(D, B)))
            ixgrid = np.ix_(ind, ind)
            k[ixgrid] += ke
        return k

    def load_vector2(self, pr, w):
        """
        assemble load vector
        :return:
        """

        def cavern_boundaries():
            """
            Calculate nodal forces on the domain's boundaries.
            """

            nind_c1 = np.array([], dtype='i')  # 1 cavern nodes indexes
            nind_c2 = np.array([], dtype='i')  # 2 cavern nodes indexes

            for i in range(len(node_ind)):
                if len(pg) == 2:
                    if ph_group[i] == pg[0]:  # 1 Cavern's wall nodes group
                        nind_c1 = np.append(nind_c1, node_ind[i, :])
                    if ph_group[i] == pg[1]:  # 2 Cavern's wall nodes group
                        nind_c2 = np.append(nind_c2, node_ind[i, :])
                elif len(pg) == 1:
                    if ph_group[i] == pg[0]:
                        nind_c1 = np.append(nind_c1, node_ind[i, :])

            nind_c1 = np.unique(nind_c1)
            nind_c2 = np.unique(nind_c2)
            alpha = np.array([])
            d = np.array([])
            nind_c = np.concatenate((nind_c1, nind_c2), axis=0)

            for i in nind_c:
                index = np.where((node_ind == i))[0]
                nindex = node_ind[index].flatten()
                seen = set([i])
                neighbours = [x for x in nindex if x not in seen and not seen.add(x)]

                if x[i] < max(x) and x[i] > min(x):
                    alpha1 = np.arctan((x[neighbours[0]] - x[i]) / (y[i] - y[neighbours[0]]))
                    alpha2 = np.arctan((x[i] - x[neighbours[1]]) / (y[neighbours[1]] - y[i]))
                    d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                    d2 = np.sqrt((x[i] - x[neighbours[1]]) ** 2 + (y[i] - y[neighbours[1]]) ** 2)
                    d = np.append(d, (d1 + d2) / 2)
                    if i in nind_c1:
                        alpha = np.append(alpha, ((alpha1 + alpha2) / 2))
                    elif i in nind_c2:
                        alpha = np.append(alpha, -np.pi + ((alpha1 + alpha2) / 2))

                elif x[i] == min(x):
                    if y[i] > 0:
                        alpha = np.append(alpha, np.pi / 2)
                        d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                        d = np.append(d, d1)
                    elif y[i] < 0:
                        alpha = np.append(alpha, -np.pi / 2)
                        d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                        d = np.append(d, d1)

                elif x[i] == max(x):
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

            px[abs(px) < 0.001] = 0
            py[abs(py) < 0.001] = 0

            return px, py, nind_c

        x, y = self.__mesh.coordinates()  # nodal coordinates
        nnodes = self.__nnodes  # number of nodes
        ph_group = self.__mesh.group()  # physical group of a line
        pg = np.unique(self.__mesh.group())
        node_ind = self.__mesh.edges()  # nodes indexes of an element's edge
        f = np.zeros((2 * nnodes, 1))
        # px, py, nind_c = cavern_boundaries()

        for elt in self.__elts:
            node = elt.nodes()
            ind = elt.dofnos()
            fe = np.zeros(6)
            j = 0
            area = elt.area()

            for i in range(3):

                # Applying Newman's B.C. on the right edge
                if x[node[i]] == np.max(x):
                    fe[j] += -1e6 * w
                j = j + 2

                # Applying Newman's B.C. on the cavern's wall (Pressure inside the cavern)
                # if node[i] in nind_c:
                #     fe[2 * i] += area /3 * px[np.where(nind_c == node[i])]
                #     fe[2 * i + 1] += area /3 * py[np.where(nind_c == node[i])]

            f[ind] = fe.reshape((6, 1))

        return f

    def creep_load_vector2(self, dt, th, a, n, q, r, temp, stress, strain_crg):
        """
        assemble creep load vector
        :return:
        """

        def deviatoric_stress():
            dstressx = stress[0] - 0.5 * (stress[0] + stress[1])
            dstressy = stress[1] - 0.5 * (stress[0] + stress[1])

            return np.array([dstressx, dstressy, stress[2]])

        def assemble_creep_forces_vector():
            fcr = np.zeros((self.__nDOFs, 1))

            for elt in self.__elts:
                area = elt.area()
                ind = elt.dofnos()
                B = self.strain_disp_matrix(elt.eltno())
                D = elt.el_tenz()
                fcre = th * area / 3 * np.dot(np.transpose(B), np.dot(D, strain_crg[:, elt.eltno()]))
                fcr[ind] += fcre.reshape((6, 1))

            return fcr

        dstressg = deviatoric_stress()
        svmg = von_mises_stress(stress)
        g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * np.exp(- q / (r * temp))
        strain_crg = strain_crg + g_crg * dt
        f_cr = assemble_creep_forces_vector()

        return f_cr, strain_crg

    def gauss_strain(self, u):
        """
        Stresses and strains evaluated at Gaussian points.
        """

        strain = np.zeros((3, self.__nele))
        for elt in self.__elts:
            node = elt.nodes()
            B = self.strain_disp_matrix(elt.eltno())
            q = np.array([u[node[0] * 2], u[node[0] * 2 + 1],
                          u[node[1] * 2], u[node[1] * 2 + 1],
                          u[node[2] * 2], u[node[2] * 2 + 1], ])
            strain[:, [elt.eltno()]] = np.dot(B, q)

        return strain

    def nodal_extrapolation(self, variable_g):
        """Stress and strains extrapolated to nodal points."""

        variable = np.zeros((3, self.__nnodes))

        for node in range(self.__nnodes):
            i, j = np.where(self.__mesh.cells() == node)
            ntri = i.shape[0]
            area = np.zeros((ntri, 1))
            variable_e = np.zeros((3, ntri))

            for ii in range(ntri):
                ind = j[ii]
                area[ii] = self.__elts[ind].area()
                variable_e[:, ii] = variable_g[:, ind]

            areat = np.sum(area)
            variable[:, [node]] = (1 / areat) * np.dot(variable_e, area)

        return variable

    def gauss_stress(self, straing):

        stressg = np.zeros((3, self.__nele))
        for elt in self.__elts:
            stressg[:, elt.eltno()] = np.dot(elt.el_tenz(), straing[:, elt.eltno()])

        return stressg


class anl(object):
    """
    Analytical solution
    """

    def __init__(self, c1, c2):
        self.__xsym, self.__ysym = sp.symbols('xsym ysym')
        self.__ux = sp.cos(c1 * np.pi * self.__xsym + c2) * sp.cos(c1 * np.pi * self.__ysym + c2)
        self.__uy = - sp.cos(c1 * np.pi * self.__xsym + c2) * sp.cos(c1 * np.pi * self.__ysym + c2)

    def evaluate_displacement(self, mesh):
        u = np.zeros((2 * mesh.nnodes(),))
        x, y = mesh.coordinates()
        u_x = sp.lambdify([self.__xsym, self.__ysym], self.__ux, "numpy")(x, y)
        u_y = sp.lambdify([self.__xsym, self.__ysym], self.__uy, "numpy")(x, y)
        u[::2] = u_x
        u[1::2] = u_y
        u = u.reshape((2 * mesh.nnodes(), 1))

        return u

    def evaluate_strain(self, mesh):
        x, y = mesh.coordinates()
        dudx = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__ux, self.__xsym), "numpy")(x, y)
        dvdy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__uy, self.__ysym), "numpy")(x, y)
        dudy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__ux, self.__ysym), "numpy")(x, y)
        dvdx = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__uy, self.__xsym), "numpy")(x, y)

        return np.concatenate(([dudx], [dvdy], [dudy + dvdx]), axis=0)

    def evaluate_forces(self, mesh, lamda, mu):
        f = np.zeros((2 * mesh.nnodes(),))
        x, y = mesh.coordinates()
        du2dx = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__ux, self.__xsym, 2), "numpy")(x, y)
        dv2dx = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__uy, self.__xsym, 2), "numpy")(x, y)
        du2dy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__ux, self.__ysym, 2), "numpy")(x, y)
        dv2dy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__uy, self.__ysym, 2), "numpy")(x, y)
        du2dxdy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__ux, self.__xsym, self.__ysym), "numpy")(x, y)
        dv2dxdy = sp.lambdify([self.__xsym, self.__ysym], sp.diff(self.__uy, self.__xsym, self.__ysym), "numpy")(x, y)
        fx = -((lamda + 2 * mu) * du2dx + lamda * dv2dxdy + mu * (du2dy + dv2dxdy))
        fy = -((lamda + 2 * mu) * dv2dy + lamda * du2dxdy + mu * (dv2dx + du2dxdy))
        f[::2] = fx
        f[1::2] = fy
        f = f.reshape((2 * mesh.nnodes(), 1))

        return f
