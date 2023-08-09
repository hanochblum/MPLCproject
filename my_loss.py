from overlap_integral import eff_overlap_integral


def my_loss(output, label):
    eff, overlapintegral = eff_overlap_integral(output, label)
    loss = 1 - eff
    return loss


