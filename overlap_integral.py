import torch


def eff_overlap_integral(E1, E2):
    A = E1 * E2
    overlapintegral = (torch.trapz(torch.trapz(A)))
    # comment for overlap integral norm.
    # E1_Energy = torch.trapz(torch.trapz(abs(E1) ** 2))
    # E2_Energy = torch.trapz(torch.trapz(abs(E2) ** 2))
    # overlapintegral = overlapintegral / (torch.sqrt(E1_Energy * E2_Energy))
    eff = abs(overlapintegral) ** 2
    return eff, overlapintegral