
import torch
from matplotlib import pyplot as plt


from def_plane import gaus_total, gaus_list, fiber_mode_list
from def_movie import def_movie
from def_param import res, vcsel_num, fiber_mode_number_list, iterations_length, run_movies_ALL_VCSELs_with_phase_mask, \
    run_movie_only_lens, eff_list, plot_epoch_vs_eff, plot_phase_masks, planes_num
from run_optimize import run_optimize_simple_training
from build_coupling_matrix import build_coupling_matrix

#######################################################
# All configuration defined at def_param.py
#######################################################


if __name__ == '__main__':

    # initial phase masks
    weights = torch.ones(planes_num, res, res, dtype=torch.cfloat, requires_grad=True)

    # MPLC training
    for iter in range(iterations_length):
        for vcsel_count in range(vcsel_num):
            input = gaus_list[vcsel_count]
            label = fiber_mode_list[fiber_mode_number_list[vcsel_count] - 1]
            weights = run_optimize_simple_training(input, label, weights, iter, vcsel_count)
        print("Iteration number ", iter + 1)

    if plot_epoch_vs_eff:
        if iterations_length > 0:
            plt.figure()
            for vcsel_cur in range(vcsel_num):
                if ((vcsel_cur + 1) % 10) == 0:
                    plt.plot(range(iterations_length), eff_list[vcsel_cur, :], label=str(vcsel_cur + 1))
                    plt.xlabel("Iterations of algorithm")
                    plt.ylabel("Single VCSEL to target mode [dB]")
            plt.title("Single VCSEL to target mode Efficiency vs Iterations")
            plt.legend()
            plt.show()

    # plot phase masks
    if plot_phase_masks:
        with torch.no_grad():
            teta1 = weights[0, :, :]
            teta2 = weights[1, :, :]
            teta3 = weights[2, :, :]
            plt.figure()
            plt.title("plane 1")
            plt.xlabel(u"\u03bcm")
            plt.ylabel(u"\u03bcm")
            # plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, teta1)
            plt.imshow(torch.real(teta1))
            plt.colorbar()
            plt.figure()
            plt.title("plane 2")
            plt.xlabel(u"\u03bcm")
            plt.ylabel(u"\u03bcm")
            # plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, teta2)
            plt.imshow(torch.real(teta2))
            plt.colorbar()
            plt.figure()
            plt.title("plane 3")
            plt.xlabel(u"\u03bcm")
            plt.ylabel(u"\u03bcm")
            # plt.pcolormesh(xs_frame * 10e5, ys_frame * 10e5, teta3)
            plt.imshow(torch.real(teta3))
            plt.colorbar()
            plt.show()


    # Coupling matrix
    build_coupling_matrix(weights, "only_lens")  # No MPLC
    build_coupling_matrix(weights, "system_simple_training")  # With MPLC

    # video with MPLC
    if run_movies_ALL_VCSELs_with_phase_mask:
        movie_type = "MPLC"
        print("Start movie ALL VCSELs MPLC...")
        arr = def_movie(gaus_total, weights, movie_type)

    if run_movie_only_lens:
        movie_type = "only_lens"
        arr = def_movie(gaus_total, weights, movie_type)