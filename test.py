import numpy as np
from classeslib import *

# Array = np.array([[1, 2, 3], [4, 5, 6]])
# x, y  = Array

mesh = Mesh('./mesh/new_cave.msh')
sfns = Shapefns()
fe = FiniteElement(mesh, sfns, 7)
j = fe.jacobi()
V = FunctionSpace(mesh, sfns)

print(str(mesh.Ndofs()))
