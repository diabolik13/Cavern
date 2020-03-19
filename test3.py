import numpy as np
import sympy as sp

x, y = sp.symbols('x y')
expr = sp.sin(x) * sp.sin(y) + sp.cos(y)
expr2 = sp.diff(expr, x)
f = sp.lambdify([x, y], expr2, 'numpy')

# I1 = np.linspace(0, 1000, 1000)
# J2 = I1
# a = np.array([1, 2])
# b = np.array([1, 2])
# print(f(I1, J2))
# Starts here:
alpha_q = 8e-4
n = 3
gamma = 0.11
beta_1 = 4.8e-3
beta = 0.995
m = -0.5
I1a, J2a, J3a = sp.symbols('I1 J2 J3')
I1 = np.linspace(0, 1000, 1000)
J2 = I1
J3 = I1
theta = (-np.sqrt(27) * J3a) / (2 * np.power(J2a, 1.5))
Fb = (-alpha_q * np.power(I1a, n) + gamma * np.power(I1a, 2))
Fs = np.power(sp.exp(beta_1 * I1a) - beta * theta, m)
Qvp = J2a - Fb * Fs
dQvp = sp.diff(Qvp, I1a)
dUvp = sp.diff(Qvp, J2a)
dVvp = sp.diff(Qvp, J3a)

sigma_t = 1.8  # Tensile strength
sigma_xx, sigma_yy, sigma_xy = sp.symbols('sigma_xx sigma_yy sigma_xy')
stress_xx = np.linspace(0, 500, 1000)
stress_yy = stress_xx
stress_xy = 0.5 * stress_xx


I1b = sigma_xx + sigma_yy + 3 * sigma_t
I2b = sigma_xx * sigma_yy - np.power(sigma_xy, 2)
J2b = (1/3) * np.power(I1b, 2) - I2b
J3b = (2/27) * np.power(I1b, 3) - (1/3) * I1b * I2b

dI1dsigma_xx = sp.diff(I1b, sigma_xx)
dJ2dsigma_xx = sp.diff(J2b, sigma_xx)
dJ3dsigma_xx = sp.diff(J3b, sigma_xx)

test = dQvp * dI1dsigma_xx
print(test)
g = sp.lambdify([I1a, J2a, J3a], test, 'numpy')
t = g(I1, J2, J3)  # very very quick function :)
# g = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], dJ2dsigma_xx, 'numpy')
print('done')