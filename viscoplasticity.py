import numpy as np
import sympy as sp
import meshio
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
import sys


# Viscoplasticity functions:
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
    test = np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])
    b = 1 / np.linalg.det(j) * np.array([[yc[1, 2], 0, yc[2, 0], 0, yc[0, 1], 0],
                                         [0, xc[2, 1], 0, xc[0, 2], 0, xc[1, 0]],
                                         [xc[2, 1], yc[1, 2], xc[0, 2], yc[2, 0], xc[1, 0], yc[0, 1]]])

    return b


def assemble_vp_force_vector(dof, p, t, d, evp, th):
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
        fvpe = th * area * np.dot(np.transpose(b), np.dot(d, evp[:, e]))
        # fvpe = (th * area * np.dot(np.transpose(b), np.dot(d, evp[:, e])))*100
        # fvpe = (th * area * np.dot(np.transpose(b), np.dot(d, evp[:, e])))/10


        for i in range(6):
            fvp[ind[i]] = fvp[ind[i]] + fvpe[i]

    return fvp


def nodal_yield_function(p, t, Fvp):
    """Yield function extrapolated to nodal points."""

    nnodes = p.shape[1]
    Fvp_nod = np.zeros(nnodes)

    for k in range(nnodes):
        i, j = np.where(t == k)
        ntri = i.shape[0]
        area = np.zeros((ntri, 1))      # maybe have to adapt this one
        fvpe = np.zeros(ntri)

        for ii in range(ntri):
            ind = j[ii]
            el = np.array([p[:, t[0, ind]], p[:, t[1, ind]], p[:, t[2, ind]]])
            x = np.array(el[:, 0])
            y = np.array(el[:, 1])
            area[ii] = polyarea(x, y)
            fvpe[ii] = Fvp[ind]

        areat = np.sum(area)
        Fvp_nod[k] = np.dot(fvpe, ((1 / areat) * area))

    return Fvp_nod


def convert_stress(stressg):
    stress_mpa = stressg * 1e-6
    return stress_mpa


def first_stress_inv(stress_mpa, sigma_t):
    I1 = np.zeros(len(stress_mpa[0]))
    for i in range(len(stress_mpa[0])):
        if stress_mpa[0, i] > 0 and stress_mpa[1, i] > 0:
            I1[i] = (stress_mpa[0, i] + stress_mpa[1, i]) + sigma_t
        if stress_mpa[0, i] < 0 and stress_mpa[1, i] < 0:
            I1[i] = -(stress_mpa[0, i] + stress_mpa[1, i]) + sigma_t
        if stress_mpa[0, i] < 0 and stress_mpa[1, i] > 0:
            I1[i] = (-stress_mpa[0, i] + stress_mpa[1, i]) + sigma_t
        if stress_mpa[0, i] > 0 and stress_mpa[1, i] < 0:
            I1[i] = (stress_mpa[0, i] - stress_mpa[1, i]) + sigma_t
    I1_check = np.where(I1 < 0)
    return I1


def second_stress_inv(stress_mpa):
    I2 = stress_mpa[0, :] * stress_mpa[1, :] - np.power(stress_mpa[2, :], 2)
    return I2


def second_dev_stress_inv(I1, I2):
    J2 = (1 / 3) * np.power(I1, 2) - I2
    J2_check = np.where(J2 < 0)
    return J2


def third_dev_stress_inv(I1, I2):
    J3 = (2 / 27) * np.power(I1, 3) - (1 / 3) * I1 * I2
    return J3


def lode_angle(stress_mpa, J2, J3):
    bracket = np.zeros(len(stress_mpa[0]))

    for j in range(len(stress_mpa[0])):
        if stress_mpa[0, j] > 0 and stress_mpa[1, j] > 0:
            bracket[j] = (np.sqrt(27) * J3[j]) / (2 * np.power(J2[j], 1.5))
        if stress_mpa[0, j] < 0 and stress_mpa[1, j] < 0:
            bracket[j] = (-np.sqrt(27) * J3[j]) / (2 * np.power(J2[j], 1.5))
        if stress_mpa[0, j] < 0 and stress_mpa[1, j] > 0:
            bracket[j] = (-np.sqrt(27) * J3[j]) / (2 * np.power(J2[j], 1.5))
        if stress_mpa[0, j] > 0 and stress_mpa[1, j] < 0:
            bracket[j] = (np.sqrt(27) * J3[j]) / (2 * np.power(J2[j], 1.5))

    invalid_low = np.where(bracket < -1)
    invalid_high = np.where(bracket > 1)
    theta = (1 / 3) * np.arccos(bracket)
    theta_degrees = theta * (180 / np.pi)  # from 0-30: extension, 30-60: compression
    return theta, theta_degrees


def hardening_param(t, straing, Fvp, alpha_1, eta, alpha_0, kv, associate):
    alpha = np.zeros(len(t[0]))
    alpha_q = np.zeros(len(t[0]))
    straing1 = np.transpose(straing)
    tot_acc_strain = np.zeros(len(t[0]))
    vol_acc_strain = np.zeros(len(t[0]))
    ksi = np.zeros(len(t[0]))
    ksi_v = np.zeros(len(t[0]))
    if associate == 0:
        for i in range(len(tot_acc_strain)):
            if Fvp[i] > 0:
                tot_acc_strain[i] = np.dot((straing1[i, :] * 100), (straing[:, i] * 100))
                ksi[i] = np.sqrt(tot_acc_strain[i])
                alpha[i] = alpha_1 / np.power(ksi[i], eta)
                alpha_q[i] = alpha[i]   # associated flow rule
            else:
                alpha[i] = 0
                alpha_q[i] = 0
    if associate == 1:
        for i in range(len(tot_acc_strain)):
            if Fvp[i] > 0:
                tot_acc_strain[i] = np.dot((straing1[i, :] * 100), (straing[:, i] * 100))
                vol_acc_strain[i] = np.dot((straing1[i, [0, 1]]), (straing[[0, 1], i]))
                ksi[i] = np.sqrt(tot_acc_strain[i])
                ksi_v[i] = np.sqrt(vol_acc_strain[i])
                alpha[i] = alpha_1 / np.power(ksi[i], eta)
                alpha_q[i] = alpha[i] + kv * (alpha_0 - alpha[i]) * (1 - (ksi_v[i] / ksi[i]))     # non associated flow rule
            else:
                alpha[i] = 0
                alpha_q[i] = 0
    return alpha, alpha_q


def yield_function(I1, J2, alpha, theta, beta_1, beta, mv, n, gamma, F0):
    Fs = np.power(np.exp(beta_1 * I1) - beta * np.cos(3 * theta), mv)       # og
    Fb = (-alpha * np.power(I1, n) + gamma * np.power(I1, 2))
    Fvp = (J2 - Fb * Fs)/100
    return Fvp


def potential_function(I1, J2, alpha_q, theta, beta_1, beta, mv, n, gamma):
    Qs = np.power(np.exp(beta_1 * I1) - beta * np.cos(3 * theta), mv)
    Qb = (-alpha_q * np.power(I1, n) + gamma * np.power(I1, 2))
    Qvp = J2 - Qb * Qs
    return Qvp


def stress_inv_der(stressxx, stressyy, stressxy, sigma_t):
    sigma_xx, sigma_yy, sigma_xy = sp.symbols('stressxx, stressyy, stressxy')
    I1 = -(sigma_xx + sigma_yy) + sigma_t
    I2 = sigma_xx * sigma_yy - np.power(sigma_xy, 2)
    J2 = (1 / 3) * np.power(I1, 2) - I2
    J3 = (2 / 27) * np.power(I1, 3) - (1 / 3) * I1 * I2
    dI1dxx = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(I1, sigma_xx), 'numpy')(stressxx, stressyy, stressxy)
    dJ2dxx = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J2, sigma_xx), 'numpy')(stressxx, stressyy, stressxy)
    dJ3dxx = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J3, sigma_xx), 'numpy')(stressxx, stressyy, stressxy)

    dI1dyy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(I1, sigma_yy), 'numpy')(stressxx, stressyy, stressxy)
    dJ2dyy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J2, sigma_yy), 'numpy')(stressxx, stressyy, stressxy)
    dJ3dyy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J3, sigma_yy), 'numpy')(stressxx, stressyy, stressxy)

    dI1dxy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(I1, sigma_xy), 'numpy')(stressxx, stressyy, stressxy)
    dJ2dxy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J2, sigma_xy), 'numpy')(stressxx, stressyy, stressxy)
    dJ3dxy = sp.lambdify([sigma_xx, sigma_yy, sigma_xy], sp.diff(J3, sigma_xy), 'numpy')(stressxx, stressyy, stressxy)
    return dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy, dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy


def pot_der(alpha_q, beta_1, beta, mv, n, gamma, I1, J2, J3):
    I1a, J2a, J3a = sp.symbols('I1 J2 J3')
    alpha_qa = sp.symbols('alpha_q')

    bracket = (-np.sqrt(27) * J3a) / (2 * np.power(J2a, 1.5))
    theta = (1 / 3) * sp.acos(bracket)

    Qs = np.power(sp.exp(beta_1 * I1a) - beta * sp.cos(3 * theta), mv)
    # Qs = np.power(sp.exp(beta_1 * I1a) + beta * sp.cos(3 * theta), mv)        # o.g.
    Qb = (-alpha_qa * np.power(I1a, n) + gamma * np.power(I1a, 2))
    Qvp = J2a - Qb * Qs

    # dQvpdI1 = sp.lambdify([I1a, J2a, J3a], sp.diff(Qvp, I1a), 'numpy')(I1, J2, J3)        # o.g.
    dQvpdI1 = sp.lambdify([I1a, J2a, J3a, alpha_qa], sp.diff(Qvp, I1a), 'numpy')(I1, J2, J3, alpha_q)
    dQvpdJ2 = sp.lambdify([I1a, J2a, J3a, alpha_qa], sp.diff(Qvp, J2a), 'numpy')(I1, J2, J3, alpha_q)
    dQvpdJ3 = sp.lambdify([I1a, J2a, J3a, alpha_qa], sp.diff(Qvp, J3a), 'numpy')(I1, J2, J3, alpha_q)
    return dQvpdI1, dQvpdJ2, dQvpdJ3


def chain_rule(dQvpdI1, dQvpdJ2, dQvpdJ3, dI1dxx, dJ2dxx, dJ3dxx, dI1dyy, dJ2dyy, dJ3dyy, dI1dxy, dJ2dxy, dJ3dxy):
    dQvpdxx = dQvpdI1 * dI1dxx + dQvpdJ2 * dJ2dxx + dQvpdJ3 * dJ3dxx
    dQvpdyy = dQvpdI1 * dI1dyy + dQvpdJ2 * dJ2dyy + dQvpdJ3 * dJ3dyy
    dQvpdxy = dQvpdI1 * dI1dxy + dQvpdJ2 * dJ2dxy + dQvpdJ3 * dJ3dxy
    return dQvpdxx, dQvpdyy, dQvpdxy


def viscoplastic_strain(t, Fvp, dQvpdxx, dQvpdyy, dQvpdxy, mu1, N1, F0):
    evp = np.zeros((3, len(t[0])))
    for i in range(len(Fvp)):
        if Fvp[i] > 0:
            evp[0, i] = mu1 * np.power(Fvp[i], N1) * dQvpdxx[i] * F0
            evp[1, i] = mu1 * np.power(Fvp[i], N1) * dQvpdyy[i] * F0
            evp[2, i] = mu1 * np.power(Fvp[i], N1) * dQvpdxy[i] * F0
        else:
            evp[:, i] = 0
    return evp



