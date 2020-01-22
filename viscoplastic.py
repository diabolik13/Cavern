import numpy as np

from elasticity import lame, load_mesh, plot_results

mesh_filename = 'gmsh.msh'  # supported formats: *.mat and *.msh
rho = 1  # rock density, [kg/m3]
K = 833.333  # Bulk modulus
mu = 384.615  # Shear modulus
P = 1  # cavern's pressure, [Pa]
Nt = 100  # number of time steps
dt = 1  # time step [s]

lamda, E, nu, D = lame(K, mu)  # calculate lame parameters, elasticity tensor etc.
m, p, t = load_mesh(mesh_filename)  # load mesh data: points and triangles

x = p[0, :]  # x-coordinates of nodes
y = p[1, :]  # y-coordinates of nodes

u = np.zeros((2, len(p[0])))
strain_x = (1e-5 * np.sin(np.pi * x)).reshape(1, len(p[0]))
strain_y = (-1e-5 * np.sin(np.pi * y)).reshape(1, len(p[0]))
strain_xy = (1e-5 * np.sin(np.pi * x) * np.cos(np.pi * x)).reshape(1, len(p[0]))
strain = np.concatenate((strain_x, strain_y, strain_xy), axis=0)
stress = np.dot(D, strain)

plot_results(p, t, u, strain, stress)

#TODO: here goes viscoplastic