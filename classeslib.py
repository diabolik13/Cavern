import meshio
import numpy as np


def polyarea(coord):
    x, y = coord
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def el_tenzor(mu, k):
    lamda = k - 2 / 3 * mu  # Elastic modulus
    d = np.array([[lamda + 2 * mu, lamda, 0],
                  [lamda, lamda + 2 * mu, 0],
                  [0, 0, mu]])
    return d


class Mesh(object):
    '''
    A mesh is a list of point global coordinates
    and a list of element definitions (by cornerpoint)
    This class loads a uniform mesh of a domain Omega
    Mesh('mesh_filename.msh')
    coordinates(n=None): return coord node n, or all coords
    cells(n=None): array of n-th cell's cornerpoint numbers, or all
    size(): returns number of elements
    '''

    def __init__(self, filename):
        '''
        N is the number of elements = number of INTERVALS
        a and b are interval endpoints
        '''
        m = meshio.read(filename)

        #  Coordinates
        self.__nodes = m.points * 1e3
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

    def Nele(self):
        return self.__N

    def Ndofs(self):
        return self.__Ndofs

    def Nnodes(self):
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

    def ph_group(self, n=None):

        if not (n is None):
            return self.__group[n]
        else:
            return self.__group

    def cell_ph_group(self, n=None):

        if not (n is None):
            return self.__pg[n]
        else:
            return self.__pg


class Shapefns(object):
    '''
    Define shape functions
    These will be defined on the local coordinates
    Shapefns()
    eval(n,xi): phi[n](xi, tau)
    ddxi(n):  dphi[n]
    ddtau(n):  dphi[n]
    size(): number of nodes for these shape functions
    '''

    def __init__(self):
        '''
        an array of functions for phi and deriv phi
        '''
        # linear shape functions
        self.__phi = [lambda xi, tau: xi,
                      lambda xi, tau: tau,
                      lambda xi, tau: 1 - xi - tau]
        # and derivatives of phi w.r.t. xi and tau
        self.__dphidxi = [1, 0, -1]
        self.__dphidtau = [0, 1, -1]
        self.__N = 3  # number of nodes in element

    def eval(self, n, xi, tau):
        '''
        the function phi[n](xi, tau), for any xi and tau
        '''
        return self.__phi[n](xi, tau)

    def ddxi(self, n):
        '''
        the function dphidxi[n]
        '''
        return self.__dphidxi[n]

    def ddtau(self, n):
        '''
        the function dphidtau[n]
        '''
        return self.__dphidtau[n]

    def size(self):
        '''
        the  number of points
        '''
        return self.__N


class FiniteElement(object):
    '''
    A single finite element
    FiniteElement(mesh,sfns,eltno,dofnos): constructor
        mesh is a Mesh
        sfns is a set of shape functions
        eltno=this element number, needs to be in mesh
        dofnos=numbers of this element's dofs (numDofs-sized array)
    endpts(): cell end points
    dofpts(): all dof locations
    dofnos(): all dof values
    numDofs(): number of dofs
    eval(n,x): phi[n](x)  (x, not xi)
    ddx(n,x):  dphi[n](x) (x, not xi)
    integral(f1=None,f2=None,derivative=False): integral(f1*f2*phi)
      f1, f2: ndof-sized vector of coeffs for local function
      derivative=True, do integral(f1*f2*dphi)
    '''

    def __init__(self, mesh, sfns, eltno, mu, kb):
        # TODO: endpts and dofpts are the same!
        '''
        mesh is the mesh it is built on
        sfns is the Shapefuns member
        eltno is this element's number
        endnos is a pair of ints giving the numbers of the endpoints
            in the mesh
        dofnos is an array of ints giving the numbers of the dofs
        '''
        assert (0 <= eltno < mesh.Nele())
        self.__eltno = eltno
        self.__endnos = mesh.cells(eltno)
        assert (len(self.__endnos) == 3)
        self.__endpts = np.array(mesh.coordinates(self.__endnos))
        self.__numDofs = 2 * sfns.size()
        dof = mesh.cells(eltno)
        self.__dofnos = np.array([2 * dof[0], 2 * dof[0] + 1, 2 * dof[1], 2 * dof[1] + 1, 2 * dof[2], 2 * dof[2] + 1])
        self.__dofpts = self.__endpts
        self.__sfns = sfns
        self.__area = polyarea(mesh.coordinates(self.__endnos))
        self.__D = el_tenzor(mu[mesh.cell_ph_group(eltno) - 1], kb[mesh.cell_ph_group(eltno) - 1])

    def eltno(self):
        ''' access element number '''
        return self.__eltno

    def el_tenz(self):
        ''' access elasticity tenzor '''
        return self.__D

    def endpts(self):
        ''' access endpoints '''
        return self.__endpts

    def nodes(self):
        ''' access dofpoints '''
        return self.__endnos

    def area(self):
        ''' evaluate area '''
        return self.__area

    def dofnos(self):
        ''' access dof point numbers '''
        return self.__dofnos

    def numDofs(self):
        ''' access numDofs '''
        return self.__numDofs

    def jacobi(self, inv=False):
        '''calculate J to perform local to global coordinates transformation'''
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
        '''
        evaluate the n-th shape function on this element
        at the spatial coordinate x
        '''

        return np.array([self.__sfns.ddxi(n), self.__sfns.ddtau(n)])


class FunctionSpace(object):
    '''
    A FunctionSpace has a list of elements
    numbered and with coords according to mesh
    FunctionSpace(mesh,sfns): constructor, sfns is ShapeFuns
    size(): number of elements
    ndofs(): number of all dofs
    dofpts(n=None): coordinates of dof[n] or all dofs
    int_phi_phi(c=None,derivative=[False,False]):
        integral(c*phi*phi) or
        integral(c*dphi*phi) or
        integral(c*dphi*dphi) or
        integral(c*phi*dphi)
    int_phi(f=None,derivative=False):
        integral(f*phi) or
        integral(f*dphi)

    '''

    def __init__(self, mesh, sfns, mu, kb):
        '''
        mesh is the mesh
        sfns is the Shapefuns
        '''
        self.__size = mesh.Nele()
        self.__nDOFs = mesh.Ndofs()
        # number the elements in same way as mesh
        self.__elts = list([])
        self.__dofpts = list([])
        for n in range(self.__size):
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
        '''
        assemble stiffness matrix
        :return:
        '''
        nDofs = self.__nDOFs
        k = np.zeros((nDofs, nDofs))
        for elt in self.__elts:
            # D = el_tenzor(mu, kb)
            D = elt.el_tenz()
            B = self.strain_disp_matrix(elt.eltno())
            # B = np.zeros((3, 2))
            ind = elt.dofnos()
            area = elt.area()
            # invj = np.linalg.inv(elt.jacobi())
            # for i in range(3):
            #     dNl = elt.derivative(i)
            #     dN = np.dot(invj, dNl)
            #     Bi = np.array([[dN[0], 0],
            #                    [0, dN[1]],
            #                    [dN[1], dN[0]]])
            #     B = np.append(B, Bi, axis=1)
            # B = np.delete(B, [0, 1], axis=1)
            ke = th * area * np.dot(B.transpose(), (np.dot(D, B)))
            ixgrid = np.ix_(ind, ind)
            k[ixgrid] += ke
        return k

    def load_vector(self, mesh, pr, w):
        '''
            assemble load vector
            :return:
        '''

        def cavern_boundaries(pr, w):
            '''
            Calculate nodal forces on the domain's boundaries.
            '''

            x = p[0]  # x-coordinates of nodes
            y = p[1]  # y-coordinates of nodes
            nind_c = np.array([], dtype='i')  # cavern nodes indexes

            for i in range(len(node_ind)):
                if ph_group[i] == 4:
                    nind_c = np.append(nind_c, node_ind[i, :])

            nind_c = np.unique(nind_c)
            alpha = np.array([])
            d = np.array([])

            for i in nind_c:
                index = np.where((node_ind == i))[0]
                nindex = node_ind[index].flatten()
                seen = set([i])
                neighbours = [x for x in nindex if x not in seen and not seen.add(x)]

                if x[i] > min(x):
                    alpha1 = np.arctan((x[neighbours[0]] - x[i]) / (y[i] - y[neighbours[0]]))
                    alpha2 = np.arctan((x[i] - x[neighbours[1]]) / (y[neighbours[1]] - y[i]))
                    d1 = np.sqrt((x[i] - x[neighbours[0]]) ** 2 + (y[i] - y[neighbours[0]]) ** 2)
                    d2 = np.sqrt((x[i] - x[neighbours[1]]) ** 2 + (y[i] - y[neighbours[1]]) ** 2)
                    d = np.append(d, (d1 + d2) / 2)
                    alpha = np.append(alpha, ((alpha1 + alpha2) / 2))

                elif x[i] == min(x):
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

        p = mesh.coordinates()
        t = mesh.cells()
        nnodes = mesh.Nnodes()  # number of nodes
        nele = mesh.Nele()  # number of elements
        ph_group = mesh.group()  # physical group of a line
        node_ind = mesh.edges()  # nodes indexes of the edge
        f = np.zeros((2 * nnodes, 1))
        px, py, nind_c = cavern_boundaries(pr, w)

        for k in range(nele):
            el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
            x = np.array(el[:, 0])
            y = np.array(el[:, 1])
            area = polyarea(p)
            node = np.array([t[0, k], t[1, k], t[2, k]])
            ind = [node[0] * 2, node[0] * 2 + 1, node[1] * 2, node[1] * 2 + 1, node[2] * 2, node[2] * 2 + 1]
            fe = np.zeros(6)
            j = 0

            for i in range(3):

                # Applying Newman's B.C. on the right edge
                # if x[i] == 1:
                #     fe[j] = -1
                # j = j + 2

                # Applying Newman's B.C. on the cavern's wall (Pressure inside the cavern)
                if node[i] in nind_c:
                    fe[2 * i] = fe[2 * i] + px[np.where(nind_c == node[i])]
                    fe[2 * i + 1] = fe[2 * i + 1] + py[np.where(nind_c == node[i])]

            for i in range(6):
                f[ind[i]] = f[ind[i]] + fe[i]

        return f

    def gauss_stress_strain(self, mesh, u):
        """Stress and strains evaluated at Gaussian points."""
        p = mesh.coordinates()
        t = mesh.cells()
        nele = mesh.Nele()
        strain = np.zeros((3, nele))
        stress = np.zeros((3, nele))
        for elt in self.__elts:
            # el = np.array([p[:, t[0, k]], p[:, t[1, k]], p[:, t[2, k]]])
            el = elt.endpts()
            # node = np.array([t[0, k], t[1, k], t[2, k]])
            node = elt.nodes()
            # b = generate_displacement_strain_matrix(el)
            B = self.strain_disp_matrix(elt.eltno())
            q = np.array([u[node[0] * 2], u[node[0] * 2 + 1],
                          u[node[1] * 2], u[node[1] * 2 + 1],
                          u[node[2] * 2], u[node[2] * 2 + 1], ])
            strain[:, [elt.eltno()]] = np.dot(B, q)
            stress[:, [elt.eltno()]] = np.dot(elt.el_tenz(), strain[:, [elt.eltno()]])

        return strain, stress
