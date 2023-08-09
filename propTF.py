from torch.fft import fftshift, fft2, ifftshift, ifft2
import torch
from math import pi

from def_param import res, WL


def gen_propTF_in_material(u1, L, z, n):
    # u1 - field, L - size of frame, z = length of propagation, n = material index
    m = len(u1)
    div_x = L / m
    fx = torch.linspace(-1 / (2 * div_x), 1 / (2 * div_x) - 1 / L, res)
    [FX, FY] = torch.meshgrid(fx, fx)

    H = torch.exp(-1j * pi * (WL / n) * z * (FX ** 2 + FY ** 2))
    H = fftshift(H)

    U1 = fft2(fftshift(u1))

    U2 = H * U1

    u2 = ifftshift(ifft2(U2))

    return u2