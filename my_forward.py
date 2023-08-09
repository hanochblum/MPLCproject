import torch
from def_param import frame_size, f_micro_lens, f_big_lens, photo_polymer, air, space_between_planes, planes_num
from def_plane import phase_plane_micro_lens, \
    phase_microlens_and_big_lens
from propTF import gen_propTF_in_material


def my_forward_simple_training(input, weights):
    plane = input
    plane = gen_propTF_in_material(plane, frame_size, f_micro_lens, 1.53)
    plane = plane * phase_plane_micro_lens
    plane = gen_propTF_in_material(plane, frame_size, space_between_planes, 1)
    for plane_counter in range(planes_num):
        teta = torch.angle(weights[plane_counter, :, :])
        mask = torch.exp(1j * teta)
        plane = plane * mask
        plane = gen_propTF_in_material(plane, frame_size, space_between_planes, 1)

    return plane

def my_forward_only_lens(input):
    plane = input
    plane = gen_propTF_in_material(plane, frame_size, f_micro_lens, photo_polymer)
    plane = plane * phase_microlens_and_big_lens
    plane = gen_propTF_in_material(plane, frame_size, f_big_lens, air)

    return plane