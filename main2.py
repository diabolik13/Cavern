from elasticity import *

# Define input parameters and load mesh
mesh_filename = 'gmsh.msh'  # supported formats: *.mat and *.msh
# rho = 2980  # rock density, [kg/m3]
rho = 1  # rock density, [kg/m3]
# K = 56.1e9  # Bulk modulus
K = 833.333  # Bulk modulus
# mu = 29.1e9  # Shear modulus
mu = 384.615  # Shear modulus
P = 1  # cavern's pressure, [Pa]
dof = 2  # degrees of freedom
Nt = 100  # number of time steps
dt = 10  # time step [s]

lamda, E, nu, D = lame(K, mu)  # calculate lame parameters, elasticity tensor etc.
m, p, t = load_mesh(mesh_filename)  # load mesh data: points and triangles

# Below are the indices of dof on the domain's boundary,
# such that L_bnd and R_bnd contain x dof indices and B_bnd
# and T_bnd contain y dof indices
L_bnd, R_bnd, B_bnd, T_bnd, D_bnd = extract_bnd(p, dof)
Px, Py, nind_c = cavern_boundaries(m, p, t, P)

# Assembling the linear system of equations: stiffness matrix k and load vector f
k = assemble_stiffness_matrix(dof, p, t, D)
f = assemble_vector(p, t, nind_c, Px, Py)

# check_matrix(k)  # uncomment to spy(k), det(k) and non-zero values

# Impose Dirichlet B.C.
k, f = impose_dirichlet(k, f, D_bnd)

# check_matrix(k)  # uncomment to spy(k), det(k) and non-zero values

# Solve system of linear equations ku = f
u = np.linalg.solve(k, f)  # nodal displacements vector

# Postprocessing for stresses and strains evaluation
straing, stressg = gauss_stress_strain(p, t, u, D)  # stress and strains evaluated at Gaussian points
# TODO: do a proper extrapolation!
strain, stress = nodal_stress_strain(p, t, straing, stressg)  # stress and strains extrapolated to nodal points

# Plot results
plot_results(p, t, u, strain, stress)

print("done")
