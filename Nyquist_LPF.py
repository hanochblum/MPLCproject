
import torch
import torch.fft as Tfft
from def_param import res, div_res


def Nyquist_LPF(origin_data):

    N_zeros = torch.zeros((res, res))
    for i in range(res):
        for j in range(res):
            if (i % div_res == 0) and (j % div_res == 0):
                N_zeros[i, j] = 1

    origin_data = origin_data * N_zeros
    fx = (torch.range(0, (res-1)) / res) - 0.5
    fy = fx
    [fxx, fyy] = torch.meshgrid(fx, fy)
    LPF = (abs(fxx) < 1 / (2 * div_res)) * (abs(fyy) < 1 / (2 * div_res))
    LPF_data = (div_res**2) * Tfft.fftshift(Tfft.ifft2(Tfft.fftshift(Tfft.fftshift(Tfft.fft2(Tfft.fftshift(origin_data))) * (LPF))))
    return torch.real(LPF_data)