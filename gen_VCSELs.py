import math
import numpy as np
import torch
from matplotlib import pyplot as plt, cm


from def_param import pitch, gaus_source_waist, res, frame_size, \
    vcsel_on_center, \
    vcsel_groups, xs_frame, ys_frame, xx_frame, yy_frame, plot_vcsels

def gen_vcsel():

    gaus_list = []
    gaus_total = torch.zeros((res, res))

    limit = (40 * 4) - 1  # change only first number
    xpos_vec = []
    ypos_vec = []
    h = math.sqrt(3) / 2 * pitch
    offset = -limit * 2 * h
    while offset < (limit*2*h):
        for i in range(-limit, limit):
            ypos_vec.append(i*(pitch/2))
            if not vcsel_on_center:
                j = -1/3 if (abs(i) % 2 != 0) else 2/3
            else:
                j = 1 if (abs(i) % 2 != 0) else 0
            xpos_vec.append(offset + j*h)
        offset += 2*h

    #  sort xpos_vec & ypos_vec
    pos_vec = []
    dict_pos_vec = {}
    for i in range(len(xpos_vec)):
        pos_vec.append((xpos_vec[i], ypos_vec[i]))
    for i in range(len(pos_vec)):
        dist = math.sqrt((pos_vec[i][0]**2 + pos_vec[i][1]**2))
        dict_pos_vec.update({(pos_vec[i][0], pos_vec[i][1]): dist})
    dict_pos_vec = {k: v for k, v in sorted(dict_pos_vec.items(), key=lambda item: item[1])}

    # take vcsels closer to center
    dist_temp = 0
    radius_count = -1
    for key_temp in dict_pos_vec.keys():
        if dict_pos_vec[key_temp] > dist_temp+0.1e-5:
            radius_count += 1
            dist_temp = dict_pos_vec[key_temp]
        if radius_count >= vcsel_groups:
            break
    for k, v in list(dict_pos_vec.items()):
        if v >= dist_temp:
            del dict_pos_vec[k]

    for key_temp in dict_pos_vec.keys():
        xpos, ypos = key_temp[0], key_temp[1]
        gaus = torch.exp(-((xx_frame - xpos) ** 2 + (yy_frame - ypos) ** 2) / (gaus_source_waist ** 2))
        Energy_gaus_norm = torch.trapz(torch.trapz(abs(gaus) ** 2))
        gaus_norm = gaus / torch.sqrt(Energy_gaus_norm)
        gaus_list.append(gaus_norm)
        gaus_total = gaus_total + gaus_norm


    micro_lens_pos_list = (res / 2) * np.ones((2, len(dict_pos_vec)))
    max_list_count = 0
    rrr2 = 0
    for key_temp in dict_pos_vec.keys():
        xpos, ypos = key_temp[0], key_temp[1]
        xpos_idx = (res / 2) * (xpos / (frame_size / 2))
        ypos_idx = (res / 2) * (ypos / (frame_size / 2))
        micro_lens_pos_list[0, max_list_count] = int((res / 2) + xpos_idx)
        micro_lens_pos_list[1, max_list_count] = int((res / 2) + ypos_idx)
        max_list_count += 1
        rrr2 += 1


    print("number VCSELs - ", len(gaus_list))
    if plot_vcsels:
        plt.figure()
        plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, abs(gaus_total.numpy()))
        plt.xlabel(u"\u03bcm")
        plt.ylabel(u"\u03bcm")
        plt.colorbar()
        plt.show()

    Energy_all_vcsels = torch.trapz(torch.trapz(abs(gaus_total) ** 2))

    return gaus_total, gaus_list, micro_lens_pos_list


# gen_vcsel()
