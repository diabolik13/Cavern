from elasticity import *
from viscoplasticity import *
from animate_plot import *

mesh_filename = 'new_cave.msh'  # supported formats: *.mat and *.msh
input_param = load_input(mesh_filename)
k = input_param['stiffness matrix']
f = input_param['external forces']
d = input_param['elasticity tensor']
p = input_param['points']
t = input_param['elements']

u = np.linalg.solve(k, f)
straing, stressg = gauss_stress_strain(p, t, u, d)
strain, stress = nodal_stress_strain(p, t, straing, stressg)

print('Maximum absolute value of dispacement is ' + str(abs(np.max(u))) + ' meters.')
