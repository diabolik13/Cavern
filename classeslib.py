import meshio
import numpy as np


def polyarea(coord):
    x, y = coord
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


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

        #  Coordinates.
        self.__nodes = m.points * 1e3
        self.__nodes = np.delete(self.__nodes, 2, axis=1)
        self.__nodes = self.__nodes.transpose()

        #  Elements.
        self.__cells = m.cells["triangle"].transpose()
        self.__N = len(self.__cells[0])

        #  Nodes.
        self.__Ndofs = len(m.points) * 2

    def size(self):
        return self.__N

    def Ndofs(self):
        return self.__Ndofs

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

        self.__phi = [lambda xi, tau: xi,
                      lambda xi, tau: tau,
                      lambda xi, tau: 1 - xi - tau]
        # and dphidxi, dphidtau (derivative of phi w.r.t. xi and tau)
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

    def __init__(self, mesh, sfns, eltno):
        # TODO: endpts and dofpts are the same!
        '''
        mesh is the mesh it is built on
        sfns is the Shapefuns member
        eltno is this element's number
        endnos is a pair of ints giving the numbers of the endpoints
            in the mesh
        dofnos is an array of ints giving the numbers of the dofs
        '''
        # this element no. is same as mesh element no.
        assert (0 <= eltno < mesh.size())
        self.__eltno = eltno
        endnos = mesh.cells(eltno)
        assert (len(endnos) == 3)
        self.__endpts = np.array(mesh.coordinates(endnos))
        self.__numDofs = 2 * sfns.size()
        # assert (sfns.size() == len(dofnos))
        self.__dofnos = mesh.cells(eltno)
        self.__dofpts = self.__endpts
        self.__sfns = sfns
        self.__area = polyarea(mesh.coordinates(endnos))
        #
        # Gauss points and weights: 3-pts are high enough for this
        #
        # self.__gausspts = np.array([1 / 3, 1 / 3])
        # self.__gausswts = np.array([1])

        # self.__gaussvals = np.empty([self.__numDofs, self.__gausspts.size])
        # for n in range(self.__numDofs):
        #     self.__gaussvals[n, :] = sfns.eval(n, self.__gausspts[:])

    def endpts(self):
        ''' access endpoints '''
        return self.__endpts

    def dofpts(self):
        ''' access dofpoints '''
        return self.__dofpts

    def area(self):
        ''' evaluate area '''
        return self.__area

    def dofnos(self):
        ''' access dof point numbers '''
        return self.__dofnos

    def numDofs(self):
        ''' access numDofs '''
        return self.__numDofs

    # def eval(self, n, x):
    #     '''
    #     evaluate the n-th shape function on this element
    #     at the spatial coordinate x
    #     '''
    #     # map x to xi
    #     xx = np.array(x)
    #     xi = (xx - self.__endpts[0]) / (self.__endpts[1] - self.__endpts[0])
    #     # evaluate
    #     return self.__sfns.eval(n, xi) * (xi >= 0.) * (xi <= 1.)
    def jacobi(self):
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

        return j

    def bmatrix(self):
        '''calculate the strain displacement matrix'''
        ddx, ddy = np.linalg.solve()

    def derivative(self, n):
        '''
        evaluate the n-th shape function on this element
        at the spatial coordinate x
        '''

        return np.array([self.__sfns.ddxi(n), self.__sfns.ddtau(n)])

    def integral(self, f1=None, f2=None, derivative=False):
        '''
        Integrate either phi[i](xi)*f1(xi)*f2(xi) or dphi[i]*f1*f2
        over this element, depending on if derivative is False or True
        Returns a vector of 3 results, one for
        phi[0], one for phi[1], and one for phi[2].
        f1 and f2 are assumed to have been mapped to this element
          as arrays
        if derivative is True, phi is replaced with dphi
        '''
        A = self.__area  # area of element
        t = self.__gausswts.copy()
        gp = self.__gausspts

        # if not (f1 is None):
        #     assert (len(f1) == self.__numDofs)
        #     fvals = np.zeros([self.__gausspts.size])
        #     for n in range(self.__numDofs):
        #         fvals += f1[n] * self.__gaussvals[n, :]
        #     t *= fvals
        #
        # if not (f2 is None):
        #     assert (len(f2) == self.__numDofs)
        #     fvals = np.zeros([self.__gausspts.size])
        #     for n in range(self.__numDofs):
        #         fvals += f2[n] * self.__gaussvals[n, :]
        #     t *= fvals

        # def derivative:
        #     # really: t *= L*(1/L)
        #     dxi = np.dot(np.array([self.__sfns.ddxi(0, gp), \
        #                            self.__sfns.ddxi(1, gp), \
        #                            self.__sfns.ddxi(2, gp)]), t)
        #
        #     dtau = np.dot(np.array([self.__sfns.ddx(0, gp), \
        #                             self.__sfns.ddx(1, gp), \
        #                             self.__sfns.ddx(2, gp)]), t)

    # def integral(self, f1=None, f2=None, derivative=False):
    #     '''
    #     Integrate either phi[i](xi)*f1(xi)*f2(xi) or dphi[i]*f1*f2
    #     over this element, depending on if derivative is False or True
    #     Returns a vector of 3 results, one for
    #     phi[0], one for phi[1], and one for phi[2].
    #     f1 and f2 are assumed to have been mapped to this element
    #       as arrays
    #     if derivative is True, phi is replaced with dphi
    #     '''
    #     L = self.__endpts[1] - self.__endpts[0]  # length of element
    #     t = self.__gausswts.copy()
    #     gp = self.__gausspts
    #
    #     if not (f1 is None):
    #         assert (len(f1) == self.__numDofs)
    #         fvals = np.zeros([self.__gausspts.size])
    #         for n in range(self.__numDofs):
    #             fvals += f1[n] * self.__gaussvals[n, :]
    #         t *= fvals
    #
    #     if not (f2 is None):
    #         assert (len(f2) == self.__numDofs)
    #         fvals = np.zeros([self.__gausspts.size])
    #         for n in range(self.__numDofs):
    #             fvals += f2[n] * self.__gaussvals[n, :]
    #         t *= fvals
    #
    #     if derivative:
    #         # really: t *= L*(1/L)
    #         q = np.dot(np.array([self.__sfns.ddx(0, gp), \
    #                              self.__sfns.ddx(1, gp), \
    #                              self.__sfns.ddx(2, gp)]), t)
    #     else:
    #         t *= L  # correct for affine map x->xi
    #         q = np.dot(np.array([self.__sfns.eval(0, gp), \
    #                              self.__sfns.eval(1, gp), \
    #                              self.__sfns.eval(2, gp)]), t)
    #
    #     return q


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

    def __init__(self, mesh, sfns):
        '''
        mesh is the mesh
        sfns is the Shapefuns
        '''
        self.__size = mesh.size()
        self.__nDOFs = mesh.Ndofs()
        # number the elements in same way as mesh
        self.__elts = list([])
        self.__dofpts = list([])
        for n in range(self.__size):
            # ASSUMING only boundary points are number 0 and (self.__size)
            # if n == 0:
            #     dofs = [2 * n, 2 * n + 1, 2 * n + 2]
            #     newdofs = range(3)
            # else:
            #     self.__nDOFs += 2
            #     dofs = [2 * n, 2 * n + 1, 2 * n + 2]
            #     newdofs = range(1, 3)
            fe = FiniteElement(mesh, sfns, n)
            self.__elts.append(fe)
            # for i in newdofs:
            #     self.__dofpts.append(fe.dofpts()[i])
        # self.__dofpts = np.array(self.__dofpts)

    def size(self):
        return len(self.__elts)

    def Ndofs(self):
        return self.__nDOFs

    # def dofpts(self, n=None):
    #     if (n is None):
    #         return self.__dofpts
    #     else:
    #         return self.__dofpts[n]

    def int_phi_phi(self, c=None, derivative=[False, False]):
        '''
        assemble $\int c(x)\phi(x)\phi(x) dx$ or with $d\phi/dx$
        '''
        A = np.zeros([self.__nDOFs, self.__nDOFs])
        # loop over elements
        for elt in self.__elts:
            d0 = elt.dofnos()
            if not (c is None):
                cc = c[d0]
            else:
                cc = None
            N = elt.numDofs()
            endpts = elt.endpts()
            Area = elt.area()  # area of elt
            for j in range(N):
                if derivative[1]:
                    # chain rule: d(xi)/d(x) = 1/L
                    phi = elt.ddx(j, elt.dofpts()) / Area
                else:
                    phi = elt.eval(j, elt.dofpts())
                A[d0, d0[j]] += elt.integral(phi, cc, derivative=derivative[0])
        return A

    def int_phi(self, f=None, derivative=False):
        '''
        assemble $\int f(x)\phi(x) dx$ or with $d\phi/dx$
        '''
        F = np.zeros(self.__nDOFs)

        for elt in self.__elts:
            d0 = elt.dofnos()
            if not (f is None):
                ff = f[d0]
            else:
                ff = None
            F[d0] += elt.integral(ff, f2=None, derivative=derivative)

        return F
