# Libraries
import meshio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sympy as sp
import sys

from elasticity import *
from CoolProp.CoolProp import PropsSI


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

    def __init__(self, filename, xfactor=1, yfactor=1, u=None):
        """
        N is the number of elements = number of INTERVALS
        a and b are interval endpoints
        """
        m = meshio.read(filename)

        #  Coordinates converted from -1...1 domain generated in Gmsh to a given scale defined by xfactor and yfactor
        m.points[:, 0] *= xfactor
        m.points[:, 1] *= yfactor

        if not (u is None):
            m.points[:, 0] += u[::2]
            m.points[:, 1] += u[1::2]

        self.__mesh = m
        self.__nodes = self.__mesh.points
        self.__nodes = np.delete(self.__nodes, 2, axis=1)
        self.__nodes = self.__nodes.transpose()

        #  Elements
        self.__cells = self.__mesh.cells["triangle"].transpose()
        self.__N = len(self.__cells[0])

        #  Nodes and dofs
        self.__Nnodes = len(self.__mesh.points)
        self.__Ndofs = self.__Nnodes * 2

        #  Physical groups
        self.__edge_pg = self.__mesh.cell_data['line']['gmsh:physical']
        self.__ele_pg = self.__mesh.cell_data["triangle"]["gmsh:physical"]

        #  Edges
        self.__edges = self.__mesh.cells['line']

        # lines and cells
        self.__cellsdata = self.__mesh.cells
        self.__pointsdata = self.__mesh.points

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
            return self.__edge_pg[n]
        else:
            return self.__edge_pg

    def edges(self, n=None):

        if not (n is None):
            return self.__edges[n]
        else:
            return self.__edges

    def line_ph_group(self, n=None):

        if not (n is None):
            return self.__edge_pg[n]
        else:
            return self.__edge_pg

    def cell_ph_group(self, n=None):

        if not (n is None):
            return self.__ele_pg[n]
        else:
            return self.__ele_pg

    def litho_bnd(self):
        t_bnd_y = np.array([], dtype='i')
        r_bnd_x = np.array([], dtype='i')
        x, y = self.__nodes
        for i in range(self.__Nnodes):
            if x[i] == np.max(x):
                r_bnd_x = np.append(r_bnd_x, i)
            if y[i] == np.max(y):
                t_bnd_y = np.append(t_bnd_y, i)
        return t_bnd_y, r_bnd_x

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
        x, y = self.__nodes

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

        # after dofs of the left, right, bottom and top edges are accessed, the function will return only the array of
        # the dofs of the edges, where dbc are chosen to be implemented by the user's choice.
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

    def update_mesh(self, u):
        """
        Updates mesh coordinates after deformation 'u', if specified by user.
        """
        self.__nodes[0] += np.transpose(u[::2].reshape((self.__Nnodes,)))
        self.__nodes[1] += np.transpose(u[::2].reshape((self.__Nnodes,)))
        self.__pointsdata[:, 0] += np.transpose(u[::2].reshape((self.__Nnodes,)))
        self.__pointsdata[:, 1] += np.transpose(u[::2].reshape((self.__Nnodes,)))

    def peak(self, sigma, xs, ys):
        """
        Function used to generate the impurities content in 2D.
        """
        x, y = self.__nodes
        f = np.exp(-((x - xs) ** 2) / sigma ** 2 - ((y - ys) ** 2) / sigma ** 2)
        return f

    def onedpeak(self, sigma, ys):
        """
        Function used to generate the impurities content in 1D.
        """
        x, y = self.__nodes
        f = np.exp(- ((y - ys) ** 2) / sigma ** 2)
        return f


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

    def __init__(self, mesh, sfns, eltno, e, nu, imp_1=None, imp_2=None):
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
        # self.__D = el_tenzor(mu[mesh.cell_ph_group(eltno) - 1], kb[mesh.cell_ph_group(eltno) - 1])
        # self.__D = el_tenzor(e, nu)
        self.__D = el_tenzor(e, nu, eltno, imp_1, imp_2)

    def create_distribution(self, mesh, plot=False):
        # Shale Young moduli and poisson ratio multivariate distribution
        xx = np.array([10e9, 24e9])
        yy = np.array([0.25, 0.35])
        means = [xx.mean(), yy.mean()]
        stds = [xx.std() / 3, yy.std() / 3]
        corr = 0.95  # correlation
        covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
                [stds[0] * stds[1] * corr, stds[1] ** 2]]
        eg, nug = np.random.multivariate_normal(means, covs, mesh.nele()).T

        # Anhydrite Young moduli and poisson ratio multivariate distribution
        xx = np.array([40e9, 58e9])
        yy = np.array([0.2, 0.3])
        means = [xx.mean(), yy.mean()]
        stds = [xx.std() / 3, yy.std() / 3]
        corr = 0.95  # correlation
        covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
                [stds[0] * stds[1] * corr, stds[1] ** 2]]
        eg2, nug2 = np.random.multivariate_normal(means, covs, mesh.nele()).T

        # Halite Young moduli and poisson ratio multivariate distribution
        xx = np.array([27e9, 58e9])
        yy = np.array([0.15, 0.43])
        means = [xx.mean(), yy.mean()]
        stds = [xx.std() / 3, yy.std() / 3]
        corr = 0.95  # correlation
        covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
                [stds[0] * stds[1] * corr, stds[1] ** 2]]
        eg3, nug3 = np.random.multivariate_normal(means, covs, mesh.nele()).T

        # Potassium salt Young moduli and poisson ratio multivariate distribution
        xx = np.array([26e9, 36e9])
        yy = np.array([0.2, 0.25])
        means = [xx.mean(), yy.mean()]
        stds = [xx.std() / 3, yy.std() / 3]
        corr = 0.95  # correlation
        covs = [[stds[0] ** 2, stds[0] * stds[1] * corr],
                [stds[0] * stds[1] * corr, stds[1] ** 2]]
        eg4, nug4 = np.random.multivariate_normal(means, covs, mesh.nele()).T

        # young modulus
        e = np.concatenate((eg, eg2, eg3, eg4))
        # poisson ration
        nu = np.concatenate((nug, nug2, nug3, nug4))

        # plot the distribution
        if plot:
            # create data
            title1 = ['shale'] * mesh.nele()
            title2 = ['anhydrite'] * mesh.nele()
            title3 = ['halite'] * mesh.nele()
            title4 = ['potash'] * mesh.nele()
            title = title1 + title2 + title3 + title4
            df = pd.DataFrame(
                {'Young modulus, [Pa]': e, 'Poisson ratio, [-]': nu, 'Rock type:': title})
            sns.lmplot(x='Young modulus, [Pa]', y='Poisson ratio, [-]', data=df, fit_reg=False, hue='Rock type:',
                       legend=False, palette="Set2", markers=[".", ".", ".", "."])
            plt.legend(loc='lower right')
            plt.show()

        return e, nu

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
        """
        calculate J to perform local to global coordinates transformation
        x,y: coordinates
        4 jacobi(inv=False): jacobian
        5 jacobi(inv=True): invariant of the jacobian
        """

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
    A FunctionSpace has a list of elements with given parameters (area, elasticity tensor etc)
    numbered and with coords according to mesh
    FunctionSpace(mesh, sfns): constructor, sfns is shape functions
    """

    def __init__(self, mesh, sfns, e, nu, imp_1=None, imp_2=None):
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
            fe = FiniteElement(mesh, sfns, n, e, nu, imp_1, imp_2)
            self.__elts.append(fe)

    def size(self):
        return len(self.__elts)

    def Ndofs(self):
        return self.__nDOFs

    def strain_disp_matrix(self, eltno):
        """
        Assemble strain-displacement matrix
        Nl: shape functions of the current element
        dNl: derivatives of the shape functions of the current element
        B: strain-displacement matrix
        """
        B = np.zeros((3, 2))
        for i in range(3):
            # access shape functions derivatives
            dNl = self.__elts[eltno].derivative(i)
            # transfer shape functions from local to global coordinates
            dN = np.dot(self.__elts[eltno].jacobi(inv=True), dNl)
            Bi = np.array([[dN[0], 0],
                           [0, dN[1]],
                           [dN[1], dN[0]]])
            # assemble strain-displacement matrix
            B = np.append(B, Bi, axis=1)
        B = np.delete(B, [0, 1], axis=1)
        return B

    def stiff_matrix(self, th=1):
        """
        assemble stiffness matrix
        nDofs: total number of degrees of freedom
        D: elasticity tenzor
        B: strain-displacement matrix
        ind: indexes of the degrees of freedom of the current element
        area: area of the current element
        ke: current element stiffness matrix
        ixgrid: indexes of the dofs of the current element stiffness matrix in the global stiffness matrix
        k: global stiffness matrix
        """
        nDofs = self.__nDOFs
        k = np.zeros((nDofs, nDofs))
        for elt in self.__elts:
            D = elt.el_tenz()
            B = self.strain_disp_matrix(elt.eltno())
            ind = elt.dofnos()
            area = abs(elt.area())
            ke = th * area * np.dot(B.transpose(), (np.dot(D, B)))
            ixgrid = np.ix_(ind, ind)
            k[ixgrid] += ke
        return k

    def load_vector(self, rho, temp, g, depth, pressure=None, boundary=None, sign=None, i=None, period=None, th=1):
        """
        assemble load vector
        x,y: nodal coordinates
        """
        x, y = self.__mesh.coordinates()
        p = rho * g  # lithostatic pressure

        if period is not None:
            if (i % period == 0) and i != 0:
                sign = -sign

        if boundary == 'cavern':
            # d - vector of lengths between nodes on the cavern's wall
            d, alpha = self.nodal_forces()
            # indexes of the nodes of the cavern(s)
            nind_c, nind_c1, nind_c2 = self.cavern_nodes_ind()
            # depths of the roof of the cavern(s)
            d_cav_top = depth + np.max(y) - np.max(y[nind_c])
            d_cav_bot = depth + np.max(y) - np.min(y[nind_c])
            # minimum and maximum allowable cavern pressures
            pc_min = 0.2 * p * d_cav_bot  # minimum allowable cavern pressure, [Pa]
            pc_max = 0.8 * p * d_cav_top  # maximum allowable cavern pressure, [Pa]

            if pressure == 'max':
                pc = pc_max
            elif pressure == 'min':
                pc = pc_min

            if sign == 1:
                pc = pc_max
            elif sign == -1:
                pc == pc_min

            # hydrogen density calculated using the coolprop library
            rho_h2 = PropsSI('D', 'T', temp, 'P', pc, 'hydrogen')

            # initialize the forces vector
            f = np.zeros((2 * self.__nnodes, 1))
            # assemble the forces vector
            for elt in self.__elts:
                node = elt.nodes()
                ind = elt.dofnos()
                fe = np.zeros(6)

                for i in range(3):
                    # Applying lithostatic Newman's B.C. on top and right edges
                    if node[i] in nind_c:
                        dp = (pc + rho_h2 * g * (np.max(y[nind_c] - y[node[i]])) - p * (
                                depth + np.max(y) - y[node[i]])) * d[np.where(nind_c == node[i])] * th
                        # dp = (pc + rho_h2 * g * (max(y[nind_c] - y[node[i]]))) * d[np.where(nind_c == node[i])] * th
                        fe[2 * i] += dp * np.cos(alpha[np.where(nind_c == node[i])])
                        fe[2 * i + 1] += dp * np.sin(alpha[np.where(nind_c == node[i])])

                f[ind] = fe.reshape((6, 1))

        elif boundary == 'right':
            f = np.zeros((2 * self.__nnodes, 1))

            for elt in self.__elts:
                node = elt.nodes()
                ind = elt.dofnos()
                fe = np.zeros(6)

                for i in range(3):
                    if x[node[i]] == np.max(x) and abs(y[node[i]]) != np.max(y) and abs(y[node[i]]) != np.min(y):
                        fe[2 * i] += -rho * g * (depth + np.max(y) - y[node[i]]) * abs(np.max(y) - np.min(y)) / 15
                    elif x[node[i]] == np.max(x) and (y[node[i]] == np.max(y) or y[node[i]] == np.min(y)):
                        fe[2 * i] += -1 / 2 * rho * g * (depth + np.max(y) - y[node[i]]) * abs(
                            np.max(y) - np.min(y)) / 15
                f[ind] = fe.reshape((6, 1))

        elif boundary == 'top':
            f = np.zeros((2 * self.__nnodes, 1))

            for elt in self.__elts:
                node = elt.nodes()
                ind = elt.dofnos()
                fe = np.zeros(6)

                for i in range(3):
                    if y[node[i]] == np.max(y) and abs(x[node[i]]) != np.max(x) and abs(x[node[i]]) != np.min(x):
                        fe[2 * i + 1] += -2 * rho * g * depth * abs(np.max(x) - np.min(x)) / 15
                    elif y[node[i]] == np.max(y) and (x[node[i]] == np.max(x) or x[node[i]] == np.min(x)):
                        fe[2 * i + 1] += -1 / 2 * 2 * rho * g * depth * abs(np.max(x) - np.min(x)) / 15
                f[ind] = fe.reshape((6, 1))

        return f, sign

    def creep_load_vector(self, dt, a, n, q, r, temp, stress, strain_crg, arrhenius=None, omega=None):
        """
        assemble creep load vector
        :return:
        """

        def deviatoric_stress():
            """
            calculate deviatoric stress
            """
            dstressx = stress[0] - 0.5 * (stress[0] + stress[1])
            dstressy = stress[1] - 0.5 * (stress[0] + stress[1])

            return np.array([dstressx, dstressy, stress[2]])

        def assemble_creep_forces_vector():
            """
            assemble the global fictitious creep forces vector
            """
            fcr = np.zeros((self.__nDOFs, 1))

            for elt in self.__elts:
                area = elt.area()
                ind = elt.dofnos()
                B = self.strain_disp_matrix(elt.eltno())
                D = elt.el_tenz()
                fcre = 1 / 2 * area * np.dot(np.transpose(B), np.dot(D, strain_crg[:, elt.eltno()]))
                fcr[ind] += fcre.reshape((6, 1))

            return fcr

        dstressg = deviatoric_stress()
        svmg = von_mises_stress(stress)
        if arrhenius is not None:
            arr = np.exp(- q / (r * temp))
        else:
            arr = 1
        if omega is None:
            g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * arr
            strain_crg = strain_crg + g_crg * dt
        else:
            g_crg = 3 / 2 * a * abs(np.power(svmg, n - 2)) * svmg * dstressg * arr
            strain_crg = (strain_crg + g_crg * dt) / ((1 - omega) ** n)

        f_cr = assemble_creep_forces_vector()

        return -f_cr, strain_crg

    def gauss_strain(self, u):
        """
        Element wise calculated Gauss strain.
        """

        strain = np.zeros((3, self.__nele))
        for elt in self.__elts:
            # create an array of nodes indexes of the current element
            node = elt.nodes()
            # assemble strain-displacement matrix for the current element
            B = self.strain_disp_matrix(elt.eltno())
            # access nodal displacement values of the current element
            q = np.array([u[node[0] * 2],
                          u[node[0] * 2 + 1],
                          u[node[1] * 2],
                          u[node[1] * 2 + 1],
                          u[node[2] * 2],
                          u[node[2] * 2 + 1]])
            # calculate strain at gaussian point within a given CST element
            strain[:, [elt.eltno()]] = -np.dot(B, q)

        return strain

    def nodal_extrapolation(self, variable_g):
        """Stress and strains extrapolated from Gauss to nodal points."""

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
        """
        Element wise calculated Gauss stresses.
        """

        stressg = np.zeros((3, self.__nele))
        for elt in self.__elts:
            stressg[:, elt.eltno()] = np.dot(elt.el_tenz(), straing[:, elt.eltno()])

        return stressg

    def cavern_nodes_ind(self):
        """
        Extract indexes of the cavern's boundary nodes.
        """
        ph_group = self.__mesh.group()  # physical group of a line
        pg = np.unique(self.__mesh.group())
        pg = pg[pg < 13]  # cavern(s) index is represented by physical group 11 (and 12 in case of 2 caverns)
        node_ind = self.__mesh.edges()
        nind_c1 = np.array([], dtype='i')  # 1 cavern nodes indexes
        nind_c2 = np.array([], dtype='i')  # 2 cavern nodes indexes

        for i in range(len(node_ind)):
            if len(pg) == 2:
                if ph_group[i] == pg[0]:  # 1st Cavern's wall nodes group
                    nind_c1 = np.append(nind_c1, node_ind[i, :])
                if ph_group[i] == pg[1]:  # 2nd Cavern's wall nodes group
                    nind_c2 = np.append(nind_c2, node_ind[i, :])
            elif len(pg) == 1:  # only 1 cavern present
                if ph_group[i] == pg[0]:
                    nind_c1 = np.append(nind_c1, node_ind[i, :])

        nind_c1 = np.unique(nind_c1)
        nind_c2 = np.unique(nind_c2)
        nind_c = np.concatenate((nind_c1, nind_c2), axis=0)

        return nind_c, nind_c1, nind_c2

    def nodal_forces(self, pr=1, w=1):
        """
        Evaluate x and y component of nodal forces on the cavern's wall
        :return:
        """
        nind_c, nind_c1, nind_c2 = self.cavern_nodes_ind()
        node_ind = self.__mesh.edges()
        x, y = self.__mesh.coordinates()
        d = np.array([])
        alpha = np.array([])

        for i in nind_c:
            index = np.where((node_ind == i))[0]
            nindex = node_ind[index].flatten()
            seen = set([i])
            neighbours = [x for x in nindex if x not in seen and not seen.add(x)]

            if x[i] < np.max(x) and x[i] > np.min(x):
                alpha1 = np.arctan((x[neighbours[0]] - x[i]) / (y[i] - y[neighbours[0]]))
                alpha2 = np.arctan((x[i] - x[neighbours[1]]) / (y[neighbours[1]] - y[i]))
                d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                d2 = np.sqrt((x[i] - x[neighbours[1]]) ** 2 + (y[i] - y[neighbours[1]]) ** 2)
                d = np.append(d, (d1 + d2) / 2)
                if i in nind_c1:
                    alpha = np.append(alpha, ((alpha1 + alpha2) / 2))
                elif i in nind_c2:
                    alpha = np.append(alpha, -np.pi + ((alpha1 + alpha2) / 2))

            elif x[i] == np.max(x) or x[i] == np.min(x):
                if y[i] > 0:
                    alpha = np.append(alpha, np.pi / 2)
                    d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                    d = np.append(d, d1)
                elif y[i] < 0:
                    alpha = np.append(alpha, -np.pi / 2)
                    d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                    d = np.append(d, d1)

        return d, alpha


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


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
