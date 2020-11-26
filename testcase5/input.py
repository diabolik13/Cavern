from classeslib import *

# physics
NR = False
arrhenius = True
damage = False
cyclic = False
impurities = False
principal_stress = False

# input parameters
pressure = 'min'  # switch between minumum 'min' and maximum 'max' pressure of the cavern
boundary = 'cavern'  # whenere to apply Neumann bc ('cavern', 'right', 'top')
cfl = 0.8
g = 9.81  # gravity constant, [m/s^2]
rho = 2250  # average rock salt, [kg/m3]
depth = 600  # depth from the surface to the salt rock top interface, [m]
a = 8.1e-28  # creep material constant, [Pa^n]
bo = 7e-22  # material parameter
n = 3.5  # creep material constant, [-]
l = 2.5  # material parameter
kk = 3  # material parameter
temp = 298  # temperature, [K]
q = 51600  # creep activation energy, [J/mol]
r = 8.314  # gas constant, [J/mol/K]
th = 1  # thickness of the domain, [m]
nt = 15  # number of time steps, [-]
dt = 1e6  # time step size, [s]
scale = 2e2  # scale the domain from -1...1 to -X...X, where X - is the scale
sign = 1  # used for cyclic load
imp_1 = None
imp_2 = None
et = [0]  # elapsed time
filename = 'new_cave2.msh'
mesh = Mesh('./mesh/' + filename, scale, scale)

if impurities == True:
    ym = [7e8, 24e9, 44e9]  # Young's modulus, [Pa] for 3 different domain zones (rock salt, potash lens, shale layer)
    nu = [0.15, 0.2, 0.3]  # Poisson ratio, [-]
else:
    ym = 44e9  # Young's modulus, [Pa] for homogeneous case
    nu = 0.3  # Poisson ratio, [-] for homogeneous case

# modeling heterogeneity
if impurities == True:
    # generates impurities content across the domain and assigns it to every fe
    imp2 = mesh.peak(100, -350, 150)
    imp3 = mesh.onedpeak(25, -100)

    imp_1 = np.zeros((mesh.nele(),))
    imp_2 = np.zeros((mesh.nele(),))

    for elt in range(mesh.nele()):
        nodes = mesh.cells(elt)
        imp_1[elt] = np.mean(imp2[nodes])
        imp_2[elt] = np.mean(imp3[nodes])