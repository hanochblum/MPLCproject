import torch
from matplotlib import pyplot as plt
from torch.fft import fftshift, fft2, ifftshift, ifft2, rfft, irfft, fftfreq, rfft2, irfft2, rfftfreq

from def_param import res, div_res, N_zeros


def gen_LPF(signal, freq_cutoff):

    freq_cutoff = 1/(2*div_res)

    pass1 = torch.abs(rfftfreq(signal.shape[-1])) < freq_cutoff
    pass2 = torch.abs(fftfreq(signal.shape[-2])) < freq_cutoff


    kernel = torch.outer(pass2, pass1)

    fft_input = rfft2(signal)

    signal_new = irfft2(fft_input * kernel, s=signal.shape[-2:])

    return signal_new