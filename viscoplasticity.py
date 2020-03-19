import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio


def stress_inv(stressg):
    i1 = -(stressg[0] + stressg[1]) + 3 * 1.8
    i2 = stressg[0] * stressg[1] - np.power(stressg[2], 2)
    j2 = (1 / 3) * np.power(i1, 2) - i2
    j3 = (2 / 27) * np.power(i1, 3) - (1 / 3) * i1 * i2

    return i1, j2, j3


def lode_angle(J2, J3):
    theta = (1 / 3) * np.arccos((-np.sqrt(27) * J3) / (2 * np.power(J2, 1.5)))
    theta_degrees = theta * (180 / np.pi)  # Should be between 0 and 60 degrees
    avg_theta = np.average(theta_degrees)  # in degrees
    if any(i > 60 for i in theta_degrees):
        print('One of the Lode Angles exceeds maximum thresh hold')
    if any(i < 0 for i in theta_degrees):
        print('One of the Lode Angles exceeds minimum thresh hold')
    return theta, theta_degrees, avg_theta


def solve_yield_function(I1, J2, alpha, n, gamma, beta, beta_1, m_v, theta):
    Gvp = np.power(np.exp(beta_1 * I1) - beta * np.cos(3 * theta), m_v)
    Fvp = J2 - (-alpha * np.power(I1, n) + gamma * np.power(I1, 2)) * Gvp  # in MPa
    return Fvp


def potential_function_chain(alpha_q, n, gamma, beta_1, beta, m, I1, J2, J3, sigma_t, stressg):
    # Creation derivatives of Qvp with respect to stress invariants:
    I1a, J2a, J3a = sp.symbols('I1 J2 J3')
    theta = (-np.sqrt(27) * J3a) / (2 * np.power(J2a, 1.5))
    Fb = (-alpha_q * np.power(I1a, n) + gamma * np.power(I1a, 2))
    Fs = np.power(sp.exp(beta_1 * I1a) - beta * theta, m)
    Qvp = J2a - Fb * Fs
    dQvp = sp.diff(Qvp, I1a)
    dUvp = sp.diff(Qvp, J2a)
    dVvp = sp.diff(Qvp, J3a)

    # Creation derivatives of stress invariants with respect to stresses:
    sigma_xx, sigma_yy, sigma_xy = sp.symbols('sigma_xx sigma_yy sigma_xy')

    I1b = sigma_xx + sigma_yy + 3 * sigma_t
    I2b = sigma_xx * sigma_yy - np.power(sigma_xy, 2)
    J2b = (1/3) * np.power(I1b, 2) - I2b
    J3b = (2/27) * np.power(I1b, 3) - (1/3) * I1b * I2b

    dI1dsigma_xx = sp.diff(I1b, sigma_xx)
    dJ2dsigma_xx = sp.diff(J2b, sigma_xx)
    dJ3dsigma_xx = sp.diff(J3b, sigma_xx)

    dI1dsigma_yy = sp.diff(I1b, sigma_yy)
    dJ2dsigma_yy = sp.diff(J2b, sigma_yy)
    dJ3dsigma_yy = sp.diff(J3b, sigma_yy)

    dI1dsigma_xy = sp.diff(I1b, sigma_xy)
    dJ2dsigma_xy = sp.diff(J2b, sigma_xy)
    dJ3dsigma_xy = sp.diff(J3b, sigma_xy)

    # Chain rule to compile derivative of Qvp with respect to stresses:
    dQvpdsigmaxx = dQvp * dI1dsigma_xx + dUvp * dJ2dsigma_xx + dVvp * dJ3dsigma_xx
    dQvpdsigmayy = dQvp * dI1dsigma_yy + dUvp * dJ2dsigma_yy + dVvp * dJ3dsigma_yy
    dQvpdsigmaxy = dQvp * dI1dsigma_xy + dUvp * dJ2dsigma_xy + dVvp * dJ3dsigma_xy

    g = sp.lambdify([I1a, J2a, J3a, sigma_xx, sigma_yy, sigma_xy], dQvpdsigmaxx, 'numpy')
    h = sp.lambdify([I1a, J2a, J3a, sigma_xx, sigma_yy, sigma_xy], dQvpdsigmayy, 'numpy')
    i = sp.lambdify([I1a, J2a, J3a, sigma_xx, sigma_yy, sigma_xy], dQvpdsigmaxy, 'numpy')
    t = g(I1, J2, J3, stressg[0], stressg[1], stressg[2])
    u = h(I1, J2, J3, stressg[0], stressg[1], stressg[2])
    v = i(I1, J2, J3, stressg[0], stressg[1], stressg[2])
    return t, u, v


def desai(mu1, Fvp, F0, N1, dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy):
    evp_xx = mu1 * np.power((Fvp / F0), N1) * dQvpdsigmaxx
    evp_yy = mu1 * np.power((Fvp / F0), N1) * dQvpdsigmayy
    evp_xy = mu1 * np.power((Fvp / F0), N1) * dQvpdsigmaxy

    evp = np.array([evp_xx, evp_yy, evp_xy])
    return evp


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def generate_displacement_strain_matrix(el):
    x = np.array(el[:, 0])  # nodal coordinates
    y = np.array(el[:, 1])  # nodal coordinates
    xc = np.zeros((3, 3))
    yc = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            xc[i, j] = x[i] - x[j]
            yc[i, j] = y[i] - y[j]

    j = [[xc[0, 2], yc[0, 2]],
         [xc[1, 2], yc[1, 2]]]
    b = 1 / np.linalg.det(j) * np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])

    return b


def assemble_vp_force_vector(dof, p, t, D, evp, th):
    nnodes = p.shape[1]  # number of nodes
    nele = len(t[0])  # number of elements
    fvp = np.zeros((dof * nnodes, 1))

    for e in range(nele):
        el = np.array([p[:, t[0, e]], p[:, t[1, e]], p[:, t[2, e]]])
        x = np.array(el[:, 0])
        y = np.array(el[:, 1])
        area = polyarea(x, y)
        global_node = np.array([t[0, e], t[1, e], t[2, e]])
        ind = [global_node[0] * 2, global_node[0] * 2 + 1, global_node[1] * 2, global_node[1] * 2 + 1,
               global_node[2] * 2, global_node[2] * 2 + 1]
        b = generate_displacement_strain_matrix(el)
        fvpe = th * area * np.dot(np.transpose(b), np.dot(D, evp[:, e]))

        for i in range(6):
            fvp[ind[i]] = fvp[ind[i]] + fvpe[i]

    return fvp