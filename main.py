import time
# import tkinter

from elasticity import *
from animate_plot import *


time_start = time.time()

nt = 0
diff_u = []
diff_strain = []
diff_stress = []
# c1 = 0.5
# c2 = - np.pi / 2
c1 = 1
c2 = 0

for i in range(4):
    mesh_filename = 'consist' + str(i + 1) + '.msh'
    input_param = load_input(mesh_filename, c1, c2)
    f = input_param['external forces']
    k = input_param['stiffness matrix']
    p = input_param['points']
    t = input_param['elements']
    d = input_param['elasticity tensor']
    lamda = input_param['Lame parameter']
    mu = input_param['shear moduli']
    nnodes = p.shape[1]
    x = p[0, :]
    y = p[1, :]

    q = np.linalg.solve(k, f).reshape((2 * nnodes, 1))
    straing, stressg = gauss_stress_strain(p, t, q, d)
    strain, stress = nodal_stress_strain(p, t, straing, stressg)
    svm = von_mises_stress(stress).reshape((nnodes, 1))
    u = np.concatenate((q[::2].reshape(nnodes, 1), q[1::2].reshape(nnodes, 1)), axis=0)
    strain = np.concatenate((strain[0], strain[1], strain[2]), axis=0).reshape((3 * nnodes, 1))
    stress = np.concatenate((stress[0], stress[1], stress[2]), axis=0).reshape((3 * nnodes, 1))

    xsym, ysym = sp.symbols('xsym ysym')
    u_x = input_param['ux']
    u_y = input_param['uy']
    ux_anl = sp.lambdify([xsym, ysym], u_x, "numpy")(x, y)
    uy_anl = sp.lambdify([xsym, ysym], u_y, "numpy")(x, y)
    u_anl = np.concatenate((ux_anl.reshape(nnodes, 1), uy_anl.reshape(nnodes, 1)), axis=0)
    strain_anl = input_param['strain_anl']
    stress_anl = input_param['stress_anl']

    # output = {
    #     'displacement': u - u_anl,
    #     'strain': strain - strain_anl,
    #     'stress': stress - stress_anl,
    #     'creep forces': f,
    #     'Von Mises stress': svm,
    #     'elapsed time': nt
    # }

    plot_results(u, u_anl, strain, strain_anl, stress, stress_anl, nnodes, x, y, t)

    diff_u.append(abs(np.max(u - u_anl)))
    diff_strain.append(abs(np.max(strain - strain_anl)))
    diff_stress.append(abs(np.max(stress - stress_anl)))

    # write_results_gif(input_param, output, 15, '.png', exaggerate=False)
    # write_results_xdmf(input_param, output)

size = np.array([128, 64, 32, 16])
order_disp = (np.log(diff_u[0]) - np.log(diff_u[-1])) / (np.log(size[0]) - np.log(size[-1]))
order_strain = (np.log(diff_strain[0]) - np.log(diff_strain[-1])) / (np.log(size[0]) - np.log(size[-1]))
order_stress = (np.log(diff_stress[0]) - np.log(diff_stress[-1])) / (np.log(size[0]) - np.log(size[-1]))

plot_difference(size, diff_u, diff_strain, diff_stress, order_disp, order_strain, order_stress)

elapsed = time.time() - time_start
print('Done in {:10.2f} seconds.'.format(elapsed))
