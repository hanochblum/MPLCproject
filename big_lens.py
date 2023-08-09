import math

import numpy as np
import torch

from def_param import res, WL, big_lens_size, f_big_lens


def gen_big_lens():

    k = 2 * np.pi / WL
    xs_big_lens = (big_lens_size / 2) * torch.linspace(-1, 1, res)
    ys_big_lens = xs_big_lens
    [xx_big_lens, yy_big_lens] = torch.meshgrid(xs_big_lens, ys_big_lens)
    rr_big_lens = torch.sqrt(xx_big_lens ** 2 + yy_big_lens ** 2)
    point_big_lens = rr_big_lens <= (big_lens_size / 2)

    func_big_lens = (-k / (2 * f_big_lens)) * ((xx_big_lens ** 2) + (yy_big_lens ** 2)) * point_big_lens
    phase_big_lens = torch.exp(1j * func_big_lens)
    delta_x = (func_big_lens * WL) / (2 * math.pi)
    delta_x = 1 / (delta_x+0.00001)
    delta_x = delta_x * point_big_lens

    return phase_big_lens, func_big_lens, delta_x