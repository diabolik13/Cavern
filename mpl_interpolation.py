from matplotlib.tri import (
    Triangulation, UniformTriRefiner, CubicTriInterpolator)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def calc_gradient(triang, p, u, v):
    x = p[0, :]  # x-coordinates of nodes
    y = p[1, :]  # y-coordinates of nodes
    triang = Triangulation(x, y)
    # triang = np.transpose(triang)
    # -----------------------------------------------------------------------------
    # Refine data - interpolates the electrical potential V
    # -----------------------------------------------------------------------------
    # refiner = UniformTriRefiner(triang)
    # tri_refi, z_test_refi = refiner.refine_field(v, subdiv=3)
    # -----------------------------------------------------------------------------
    # Computes the gradient of function v
    # -----------------------------------------------------------------------------
    tci1 = CubicTriInterpolator(triang, u)
    tci2 = CubicTriInterpolator(triang, v)
    # Gradient requested here at the mesh nodes but could be anywhere else:
    ex = tci1.gradient(triang.x, triang.y)
    ey = tci2.gradient(triang.x, triang.y)

    return ex, ey