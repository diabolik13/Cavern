import numpy as np
import matplotlib.pyplot as pyplt
import sympy as sp

# Khaledi (2016)
# Material constants:
# alpha = np.array([0, 0.0002, 0.0005, 0.0008, 0.0016])
alpha = np.array([0, 0.0003, 0.0007, 0.0012, 0.002])  # values guessed to confirm Khaledi 2016
gamma = 0.11
beta_1 = 4.8e-3
beta = 0.995
m = -0.5
n = 3

# Stress:
sigma_xx = np.linspace(0, 60, 50)
sigma_yy = sigma_xx
# sigma_xy = set a value ...

# stress invariants
I1 = sigma_xx + sigma_yy  # did not take tensile strenght into account..
# I2 = sigma_xx * sigma_yy - np.power(sigma_xy, 2)
J2 = np.zeros((5, 50))
sqrt_J2 = np.zeros((5, 50))
for i in range(5):
    J2[i] = (-alpha[i] * np.power(I1, n) + gamma * np.power(I1, 2)) * np.power((np.exp(beta_1 * I1) - beta), m)
    J2[J2 < 0] = 0
    sqrt_J2 = np.sqrt(J2)

# Test case:
I1T = np.array([20, 40, 60, 80, 100, 120])
J2T = np.array([25, 100, 400, 625, 900, 1225])
sqt_T = np.sqrt(J2T)
alphat = 0
FvpT = np.zeros(6)
for j in range(6):
    FvpT[j] = J2T[j] - (alphat * np.power(I1T[j], n) + gamma * np.power(I1T[j], 2)) * np.power((np.exp(beta_1 * I1T[j]) - beta), m)

CD_boundary = (1 - (2/n)) * gamma * np.power(I1, 2) * np.power((np.exp(beta_1 * I1) - beta), m)
sqrt_CD = np.sqrt(CD_boundary)

# J2Def = (1/3) * np.power(I1, 2) - I2
# Test
pyplt.figure(1)
pyplt.plot(I1, sqrt_J2[0], '-o')
pyplt.plot(I1T, sqt_T, 'o')
pyplt.plot(I1, sqrt_J2[1], '-o')
pyplt.plot(I1, sqrt_J2[2], '-o')
pyplt.plot(I1, sqrt_J2[3], '-o')
pyplt.plot(I1, sqrt_J2[4], '-o')
# pyplt.plot(I1, sqrt_CD, '-o')
pyplt.xlabel('I1 [MPa]')
pyplt.ylabel('sqrt(J2) [MPa]')
pyplt.title('Evolution of viscoplastic yield function (Fvp) for theta = 0')
pyplt.show()
print('done')