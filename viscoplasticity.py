import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio


def stress_inv(stressg):
    i1 = -(stressg[0] + stressg[1]) + 3 * 1.8
    i2 = stressg[0] * stressg[1] - np.power(stressg[2], 2)
    j2 = (1 / 3) * np.power(i1, 2) - i2
    j3 = (2 / 27) * np.power(i1, 3) - (1 / 3) * i1 * i2  # removed I3 as its always zero

    return i1, j2, j3


def solve_yield_function(I1, J2, J3, alpha, n, gamma, beta, beta_1, m_v):
    theta = (1/3) * np.arccos((-np.sqrt(27) * J3) / (2 * np.power(J2, 1.5)))
    theta_degrees = theta * (180/np.pi)  # Should be between 0 and 60 degrees
    if any(i > 60 for i in theta_degrees):
        print('One of the Lode Angles exceeds maximum thresh hold')
    if any(i < 0 for i in theta_degrees):
        print('One of the Lode Angles exceeds minimum thresh hold')

    Gvp = np.power(np.exp(beta_1 * I1) - beta * np.cos(3 * theta), m_v)
    Fvp = J2 - (-alpha * np.power(I1, n) + gamma * np.power(I1, 2)) * Gvp  # in MPa
    return Fvp


def pot_derivatives(t, alpha_q, n, gamma, beta_1, m_v, beta, I1, J2, J3):
    alpha_q_n = sp.symbols('alpha_q_n')
    n_n = sp.symbols('n_n')
    gamma_n = sp.symbols('gamma_n')
    beta_1_n = sp.symbols('beta_1_n')
    m_v_n = sp.symbols('m_v_n')
    beta_n = sp.symbols('beta_n')
    I1_n = sp.symbols('I1_n')
    J2_n = sp.symbols('J2_n')
    J3_n = sp.symbols('J3_n')

    Qvp = J2_n - (-alpha_q_n * np.power(I1_n, n_n) + gamma_n * np.power(I1_n, 2)) * np.power(sp.exp(beta_1_n * I1_n) -
                                                                                 beta_n * (-np.sqrt(27) * J3_n) /
                                                                                 (2 * np.power(J2_n, 1.5)), m_v_n)

    dqvpdi1_n = sp.diff(Qvp, I1_n)
    dqvpdj2_n = sp.diff(Qvp, J2_n)
    dqvpdj3_n = sp.diff(Qvp, J3_n)

    nnodes = len(t[0])
    dQvpdI1 = np.zeros(nnodes)
    dQvpdJ2 = np.zeros(nnodes)
    dQvpdJ3 = np.zeros(nnodes)

    for i in range(nnodes):
        dqvpdi1 = dqvpdi1_n.subs(
            {alpha_q_n: alpha_q, n_n: n, gamma_n: gamma, beta_1_n: beta_1, m_v_n: m_v, beta_n: beta, I1_n: I1[i],
             J2_n: J2[i], J3_n: J3[i]})
        dQvpdI1[i] = dqvpdi1
        dqvpdj2 = dqvpdj2_n.subs(
            {alpha_q_n: alpha_q, n_n: n, gamma_n: gamma, beta_1_n: beta_1, m_v_n: m_v, beta_n: beta, I1_n: I1[i],
             J2_n: J2[i], J3_n: J3[i]})
        dQvpdJ2[i] = dqvpdj2
        dqvpdj3 = dqvpdj3_n.subs(
            {alpha_q_n: alpha_q, n_n: n, gamma_n: gamma, beta_1_n: beta_1, m_v_n: m_v, beta_n: beta, I1_n: I1[i],
             J2_n: J2[i], J3_n: J3[i]})
        dQvpdJ3[i] = dqvpdj3

    return dQvpdI1, dQvpdJ2, dQvpdJ3


def der_stress_inv(t, stressg):
    stress_xx = sp.symbols('stress_xx')
    stress_yy = sp.symbols('stress_yy')
    stress_xy = sp.symbols('stress_xy')
    i1 = stress_xx + stress_yy
    i2 = stress_xx * stress_yy - np.power(stress_xy, 2)
    i3 = 0
    j2 = (1 / 3) * np.power(i1, 2) - i2
    j3 = (2 / 27) * np.power(i1, 3) - (1 / 3) * i1 * i2 + i3

    # First order derivatives:
    di1dsigmaxx_n = sp.diff(i1, stress_xx)
    dj2dsigmaxx_n = sp.diff(j2, stress_xx)
    dj3dsigmaxx_n = sp.diff(j3, stress_xx)

    di1dsigmayy_n = sp.diff(i1, stress_yy)
    dj2dsigmayy_n = sp.diff(j2, stress_yy)
    dj3dsigmayy_n = sp.diff(j3, stress_yy)

    di1dsigmaxy_n = sp.diff(i1, stress_xy)
    dj2dsigmaxy_n = sp.diff(j2, stress_xy)
    dj3dsigmaxy_n = sp.diff(j3, stress_xy)

    nnodes = len(t[0])
    dI1dsigmaxx = np.zeros(nnodes)
    dJ2dsigmaxx = np.zeros(nnodes)
    dJ3dsigmaxx = np.zeros(nnodes)

    dI1dsigmayy = np.zeros(nnodes)
    dJ2dsigmayy = np.zeros(nnodes)
    dJ3dsigmayy = np.zeros(nnodes)

    dI1dsigmaxy = np.zeros(nnodes)
    dJ2dsigmaxy = np.zeros(nnodes)
    dJ3dsigmaxy = np.zeros(nnodes)

    for i in range(nnodes):
        di1dsigmaxx = di1dsigmaxx_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dI1dsigmaxx[i] = di1dsigmaxx
        dj2dsigmaxx = dj2dsigmaxx_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ2dsigmaxx[i] = dj2dsigmaxx
        dj3dsigmaxx = dj3dsigmaxx_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ3dsigmaxx[i] = dj3dsigmaxx
        di1dsigmayy = di1dsigmayy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dI1dsigmayy[i] = di1dsigmayy
        dj2dsigmayy = dj2dsigmayy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ2dsigmayy[i] = dj2dsigmayy
        dj3dsigmayy = dj3dsigmayy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ3dsigmayy[i] = dj3dsigmayy
        di1dsigmaxy = di1dsigmaxy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dI1dsigmaxy[i] = di1dsigmaxy
        dj2dsigmaxy = dj2dsigmaxy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ2dsigmaxy[i] = dj2dsigmaxy
        dj3dsigmaxy = dj3dsigmaxy_n.subs({stress_xx: stressg[0, i], stress_yy: stressg[1, i], stress_xy: stressg[2, i]})
        dJ3dsigmaxy[i] = dj3dsigmaxy

    return dI1dsigmaxx, dJ2dsigmaxx, dJ3dsigmaxx, dI1dsigmayy, dJ2dsigmayy, dJ3dsigmayy, dI1dsigmaxy, dJ2dsigmaxy, \
           dJ3dsigmaxy


def potential_stress(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dsigmaxx, dI1dsigmayy, dI1dsigmaxy, dJ2dsigmaxx, dJ2dsigmayy,
                 dJ2dsigmaxy, dJ3dsigmaxx, dJ3dsigmayy, dJ3dsigmaxy):
    dQvpdsigmaxx = dQvpdI1 * dI1dsigmaxx + dQvpdJ2 * dJ2dsigmaxx + dQvpdJ3 * dJ3dsigmaxx
    dQvpdsigmayy = dQvpdI1 * dI1dsigmayy + dQvpdJ2 * dJ2dsigmayy + dQvpdJ3 * dJ3dsigmayy
    dQvpdsigmaxy = dQvpdI1 * dI1dsigmaxy + dQvpdJ2 * dJ2dsigmaxy + dQvpdJ3 * dJ3dsigmaxy
    return dQvpdsigmaxx, dQvpdsigmayy, dQvpdsigmaxy


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