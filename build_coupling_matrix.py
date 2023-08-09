import math
import matplotlib.pyplot as plt
import torch
from def_param import vcsel_num, plot_coupling_matrix, plot_IL_each_vcsel, plot_VCSEL_to_MODE_IL
from def_plane import gaus_list, modes_to_calc, fiber_mode_list, \
    coupling_matrix_eff, coupling_matrix_overlap
from my_forward import my_forward_simple_training, my_forward_only_lens
from overlap_integral import eff_overlap_integral


def build_coupling_matrix(weights, system_type):

        with torch.no_grad():
            for vcsel_count in range(vcsel_num):
                plane = gaus_list[vcsel_count]
                if system_type == "system_simple_training":
                        plane = my_forward_simple_training(plane, weights)
                if system_type == "only_lens":
                    plane = my_forward_only_lens(plane)
                for fiber_mode_count in range(modes_to_calc):
                    fiber_mode_cur = fiber_mode_list[fiber_mode_count]
                    eff, overlapintegral = eff_overlap_integral(plane, fiber_mode_cur)
                    coupling_matrix_eff[vcsel_count, fiber_mode_count] = eff
                    coupling_matrix_overlap[vcsel_count, fiber_mode_count] = overlapintegral

        u, s, v = torch.svd(coupling_matrix_overlap)
        svd_eff_final = torch.mean(abs(s)**2).item()
        svd_eff_dB_final = 10 * math.log10(svd_eff_final)
        print("svd_eff_dB_final", svd_eff_dB_final)

        sum_eff_all_vcsels = 0
        for i in range(vcsel_num):
            sum_eff_vcsel = sum(coupling_matrix_eff[i, :])
            if plot_IL_each_vcsel:
                print("MPLC, IL of vcsel "+str(i+1)+" [dB] " + str(round(10*math.log10(sum_eff_vcsel.item()), 3)))
            if plot_VCSEL_to_MODE_IL:
                print("Single VCSEL "+str(i+1)+" to single MODE "+str(i+1)+" IL "+str(round(coupling_matrix_eff[i, i].item(), 3)))
            sum_eff_all_vcsels += (sum_eff_vcsel / vcsel_num)
        eff_final = 10 * math.log10(sum_eff_all_vcsels)
        print("IL Total [dB] = ", eff_final)
        coupling_matrix_dB = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.float64)
        for i in range(vcsel_num):
            for j in range(modes_to_calc):
                coupling_matrix_dB[i, j] = 10*math.log10(coupling_matrix_eff[i, j])


        if plot_coupling_matrix:
            plt.figure()
            plt.imshow(coupling_matrix_dB, cmap=plt.get_cmap('winter'), vmin=0, vmax=-50)
            plt.title("coupling matrix " + system_type + " IL [dB]")
            # plt.title("coupling matrix - IL [dB]")
            plt.colorbar()
            plt.ylabel("VCSELs")
            plt.xlabel("Fiber modes")
            plt.show()
