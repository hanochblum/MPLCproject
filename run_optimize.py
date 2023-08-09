from def_param import lr, eff_list
from my_forward import my_forward_simple_training
from my_loss import my_loss
import torch
import math

def run_optimize_simple_training(input, label, weights, iter, vcsel_count):

    output = my_forward_simple_training(input, weights)

    loss = my_loss(output, label)

    eff = 1 - loss.item()

    loss.backward()


    booster = (loss*3)**4  # use feedback from the loss function to generate adaptive learning rate.
    # booster = 1 # constant learning rate
    with torch.no_grad():
        weights -= weights.grad * lr * booster
        weights.grad.zero_()

    if eff > 0:
        eff_list[vcsel_count, iter] = 10 * math.log10(eff)
    return weights