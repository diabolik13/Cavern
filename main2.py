from elasticity import *

# Define input parameters and load mesh
mesh_filename = 'cave2.msh'  # supported formats: *.mat and *.msh
rho = 2980  # rock density
K = 56.1e9  # Bulk modulus
mu = 29.1e9  # Shear modulus
P = 1  # cavern's pressure
dof = 2  # degrees of freedom

lamda, E, nu, D = lame(K, mu)  # calculate lame parameters, elasticity tensor etc.
m, p, t = load_mesh(mesh_filename)  # load mesh data: points and triangles

# Below are the indices of dof on the domain's boundary,
# such that L_bnd and R_bnd contain x dof indices and B_bnd
# and T_bnd contain y dof indices
L_bnd, R_bnd, B_bnd, T_bnd = extract_bnd(p, dof)
bnd = gmsh_boundaries(m)

# Assembling the linear system of equations: stiffness matrix k and load vector f
k = assemble_stiffness_matrix(dof, p, t, D)
f = assemble_vector(p, t)

# check_matrix(k)  # uncomment to spy(k), det(k) and non-zero values

# Impose Dirichlet B.C.
D_bnd = np.concatenate((B_bnd, T_bnd, L_bnd))  # DBC on B, T and L domain edges
k[D_bnd, :] = 0
k[:, D_bnd] = 0
k[D_bnd, D_bnd] = 1
f[D_bnd] = 0

# check_matrix(k)  # uncomment to spy(k), det(k) and non-zero values

# Solve system of linear equations ku = f
u = np.linalg.solve(k, f)  # nodal displacements vector

# Postprocessing for stresses and strains evaluation
straing, stressg = gauss_stress_strain(p, t, u, D)  # stress and strains evaluated at Gaussian points
strain, stress = nodal_stress_strain(p, t, straing, stressg)  # stress and strains extrapolated to nodal points

# Plot results
plot_results(p, t, u, strain, stress)

print("done")
