import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import jv
from scipy.special import yn
from def_param import d_clad, res, xx_frame, yy_frame, xs_frame, ys_frame, \
    frame_size, v, sort_mat, matrix_of_bessel_zeros, a_core, a_clad, rr_frame, plot_fiber_mode


def gen_fiber_mode(mode_number):

    n = int(sort_mat[1, mode_number-1])
    m = int(sort_mat[2, mode_number-1])

    if n == 0 and m == 1:
        u = 2.405
    elif n == 0 and m != 1:
        u = matrix_of_bessel_zeros[0, m - 1]
    else:
        u = matrix_of_bessel_zeros[n, m - 1]

    w = np.sqrt(abs(v ** 2 - u ** 2))


    xs_clad = d_clad / 2 * torch.linspace(-1, 1, int((d_clad/frame_size)*res))
    ys_clad = xs_clad
    [xx_clad, yy_clad] = torch.meshgrid(xs_clad, ys_clad)
    rr_clad = np.sqrt(xx_clad ** 2 + yy_clad ** 2)


    teta = np.arctan2(yy_frame, xx_frame)
    point_core = rr_frame < a_core
    point_clad = ~ point_core
    u = f"{u:.5f}"
    u = float(u)
    psic = jv(n, (u * rr_frame / a_core)) * point_core / jv(n, u)
    psig = yn(n, (w * rr_frame / a_core)) * point_clad / yn(n, w)


    #### To delete ####
    psig[torch.isnan(psig)] = 0
    ###################
    plane_fiber_mode = torch.zeros((res, res), dtype=torch.float64)

    matrix_of_mode = psic + psig

    COS = np.cos(n * teta)
    SIN = np.sin(n * teta)

    smin = torch.min(matrix_of_mode)
    smax = matrix_of_mode.max()
    if (abs(smin) > smax):
        temp_smax = smax
        smax = smin
        smin = temp_smax

    ang = torch.arange(0, 2 * np.pi, 0.01)
    x_circ_core = a_core * np.cos(ang)
    y_circ_core = a_core * np.sin(ang)
    x_circ_clad = a_clad * np.cos(ang)
    y_circ_clad = a_clad * np.sin(ang)

    if n == 0:

        matrix_of_mode_normalized = matrix_of_mode  ####################################
    else:

        if (sort_mat[0, mode_number] == sort_mat[0, mode_number - 1]):
            matrix_of_mode_normalized = matrix_of_mode * COS

        else:

            matrix_of_mode_normalized = matrix_of_mode * SIN


    plane_area = torch.zeros((res, res), dtype=torch.float64)

    Energy_fiber_mode = torch.trapz(torch.trapz(abs(matrix_of_mode_normalized) ** 2))
    matrix_of_mode_normalized = matrix_of_mode_normalized / torch.sqrt(Energy_fiber_mode)
    if plot_fiber_mode:
        plt.figure()
        plt.pcolormesh(xs_frame, ys_frame, matrix_of_mode_normalized, shading='auto')
        plt.xlabel("y[" + u"\u03bcm" + "]")
        plt.ylabel(u"\u03bcm")
        # plt.imshow(abs(matrix_of_mode_normalized) ** 2)
        plt.title('LP_' + str(n) + ',' + str(m))
        plt.colorbar()
        plt.plot(x_circ_core, y_circ_core, 'black')
        plt.plot(x_circ_clad, y_circ_clad, 'black')
        plt.show()

    return matrix_of_mode_normalized, plane_area