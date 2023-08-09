import math

import numpy as np
import torch
from matplotlib import pyplot as plt

from def_param import WL, res, frame_size, f_micro_lens, micro_lens_size, \
    plot_micro_lens, xs_frame, ys_frame

def gen_micro_lens(max_list):

    #build single micro-lens
    k = 2 * np.pi / WL
    micro_lens_res = int((micro_lens_size / frame_size) * res)
    xs_micro_lens = micro_lens_size / 2 * torch.linspace(-1, 1, micro_lens_res)
    ys_micro_lens = xs_micro_lens
    [xx_micro_lens, yy_micro_lens] = torch.meshgrid(xs_micro_lens, ys_micro_lens)
    rr_micro_lens = torch.sqrt(xx_micro_lens ** 2 + yy_micro_lens ** 2)
    point_micro_lens = rr_micro_lens <= (micro_lens_size / 2)
    func_each_micro_lens = (-k / (2 * f_micro_lens)) * ((xx_micro_lens ** 2) + (yy_micro_lens ** 2)) * point_micro_lens
    smin = torch.min(torch.min(func_each_micro_lens))

    #build all micro lens
    func_plane_micro_lens = torch.zeros((res, res), dtype=torch.float64)

    if micro_lens_res % 2 == 0:
        left_idx = int((micro_lens_res) / 2)
        right_idx = int((micro_lens_res) / 2)
    else:
        left_idx = int((micro_lens_res+1) / 2)
        right_idx = int((micro_lens_res) / 2)
    for vcsel_count in range(len(max_list[0])):
        cur_pxl_i, cur_pxl_j = int(max_list[0, vcsel_count]), int(max_list[1, vcsel_count])
        test_plane_func = torch.zeros((res, res), dtype=torch.float64)
        test_plane_func[cur_pxl_i-left_idx:cur_pxl_i+right_idx,cur_pxl_j-left_idx:cur_pxl_j+right_idx] = func_each_micro_lens
        func_plane_micro_lens += test_plane_func

        delta_x2 = (func_plane_micro_lens * WL) / (2 * math.pi)
        delta_x2 = 1 / (delta_x2 + 0.00001)

    phase_plane_micro_lens = torch.exp(1j*func_plane_micro_lens)

    plane_func_micro_lens_structure = func_plane_micro_lens
    plane_func_micro_lens_structure[plane_func_micro_lens_structure == 0] = smin

    if plot_micro_lens:
        plt.figure()
        # plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, torch.angle(torch.exp(1j * plane_func_micro_lens_structure)))
        plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, plane_func_micro_lens_structure)
        plt.xlabel(u"\u03bcm")
        plt.ylabel(u"\u03bcm")
        plt.title("phase micro lens structure")
        plt.colorbar()
        plt.show()

    return phase_plane_micro_lens, func_plane_micro_lens, delta_x2, plane_func_micro_lens_structure