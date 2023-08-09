
import torch

from def_param import div_frame, res, plane_delay, frame_size, \
    f_micro_lens, f_big_lens, vcsel_num, space_between_lens, \
    run_movie_only_lens, run_movies_ALL_VCSELs_with_phase_mask, space_between_planes, \
    air, planes_num, photo_polymer
from def_plane import func_microlens_and_big_lens, phase_microlens_and_big_lens
from propTF import gen_propTF_in_material



def gen_movie_MPLC_intensity(gaus_list, weights):  # MPLC - All vcsels together

    arr_total = []
    plane_total = torch.zeros((res, res), dtype=torch.cfloat)
    dz_between_phase_masks = space_between_planes / div_frame
    dz_micro_lens = f_micro_lens / div_frame
    # suprtposition micro-lens and big lens

    print("Start movie suprtposition lenses MPLC...")

    # source
    frames_count = 0
    while (dz_micro_lens * frames_count) < f_micro_lens:
        for vcsel_count in range(vcsel_num):
            plane = gaus_list[vcsel_count]
            plane = gen_propTF_in_material(plane, frame_size, dz_micro_lens * frames_count, 1.53)
            plane_total += abs(plane) ** 2
        arr_total.append(abs(plane_total.numpy()))
        plane_total = torch.zeros((res, res), dtype=torch.cfloat)
        frames_count += 1

    # micro-micro_lens_and_big_lens_superposition
    micro_lens_idx_start = len(arr_total)
    for _ in range(plane_delay):
        arr_total.append(func_microlens_and_big_lens.numpy())

    print("Movie - done with superposition lenses...")

    # superposition lens to first plane
    frames_count = 0
    while (dz_between_phase_masks * frames_count) < space_between_planes:
        for vcsel_count in range(vcsel_num):
            plane = gaus_list[vcsel_count]
            plane = gen_propTF_in_material(plane, frame_size, f_micro_lens, 1.53)
            plane = plane * phase_microlens_and_big_lens
            plane = gen_propTF_in_material(plane, frame_size, dz_between_phase_masks * frames_count, air)
            plane_total += abs(plane) ** 2
        arr_total.append(abs(plane_total.numpy()))
        plane_total = torch.zeros((res, res), dtype=torch.cfloat)
        frames_count += 1

    # phase masks
    with torch.no_grad():
        for plane_count in range(planes_num):
            teta = torch.angle(weights[plane_count])
            for _ in range(plane_delay):
                arr_total.append(teta.numpy())
            mask = torch.exp(1j * teta)
            frames_count = 0
            while (dz_between_phase_masks * frames_count) < space_between_planes:
                plane_total = torch.zeros((res, res), dtype=torch.cfloat)
                for vcsel_count in range(vcsel_num):
                    plane = gaus_list[vcsel_count]
                    plane = gen_propTF_in_material(plane, frame_size, f_micro_lens, 1.53)
                    plane = plane * phase_microlens_and_big_lens
                    plane = gen_propTF_in_material(plane, frame_size, space_between_planes, air)
                    for i in range(plane_count):
                        teta_temp = torch.angle(weights[i])
                        mask_temp = torch.exp(1j * teta_temp)
                        plane = plane * mask_temp
                        plane = gen_propTF_in_material(plane, frame_size, space_between_planes, air)
                    plane = plane * mask
                    plane = gen_propTF_in_material(plane, frame_size, dz_between_phase_masks * frames_count, air)
                    plane_total += abs(plane) ** 2
                arr_total.append(abs(plane_total.numpy()))
                # plane_total = torch.zeros((res, res), dtype=torch.cfloat)
                frames_count += 1

    print("Done movie superposition lenses with phase mask")


    return arr_total, micro_lens_idx_start


def gen_movie_MPLC_intensity_no_lens(gaus_total, gaus_list, weights):
    arr_total = []
    dz1 = space_between_planes / div_frame
    dz2 = space_between_planes / div_frame
    plane_total = torch.zeros((res, res), dtype=torch.cfloat)
    # couping_matrix_with_phase_mask = np.zeros((vcsel_num, modes_to_calc))

    if run_movies_ALL_VCSELs_with_phase_mask:

        # source
        for _ in range(plane_delay):
            arr_total.append(gaus_total.numpy())

        # source to first plane
        frames_count = 0
        while (dz1 * frames_count) < space_between_planes:
            for vcsel_count in range(vcsel_num):
                plane = gaus_list[vcsel_count]
                plane = gen_propTF_in_material(plane, frame_size, dz1 * frames_count, air)
                plane_total += abs(plane) ** 2
            arr_total.append(abs(plane_total.numpy()))
            plane_total = torch.zeros((res, res), dtype=torch.cfloat)
            frames_count += 1

        # phase masks
        with torch.no_grad():
            for plane_count in range(planes_num):
                teta = torch.angle(weights[plane_count])
                for _ in range(plane_delay):
                    arr_total.append(teta.numpy())
                mask = torch.exp(-1j * teta)
                frames_count = 0
                while (dz2 * frames_count) < space_between_planes:
                    plane_total = torch.zeros((res, res), dtype=torch.cfloat)
                    for vcsel_count in range(vcsel_num):
                        plane = gaus_list[vcsel_count]
                        plane = gen_propTF_in_material(plane, frame_size, space_between_planes, air)
                        for i in range(plane_count):
                            teta = torch.angle(weights[i])
                            mask = torch.exp(-1j * teta)
                            plane = plane * mask
                            plane = gen_propTF_in_material(plane, frame_size, space_between_planes, air)
                        plane = plane * mask
                        plane = gen_propTF_in_material(plane, frame_size, dz2 * frames_count, air)
                        plane_total += abs(plane) ** 2
                    arr_total.append(abs(plane_total.numpy()))
                    # plane_total = torch.zeros((res, res), dtype=torch.cfloat)
                    frames_count += 1

        print("Done movie ALL VCSELs with phase mask")
    return arr_total


def gen_movie_MPLC_intensity_no_phase_masks(gaus_total, gaus_list):
    arr_total = []
    dz_micro_lens = f_micro_lens / div_frame
    dz_between_lens = space_between_lens / div_frame
    dz_big_lens = f_big_lens / div_frame
    plane_total = torch.zeros((res, res), dtype=torch.cfloat)

    if run_movie_only_lens:

# micro_lens_and_big_lens_superposition
        print("Start movie  - micro_lens_and_big_lens_superposition only lens...")

        # source to micro-lens
        frames_count = 0
        while (dz_micro_lens * frames_count) < f_micro_lens:
            for vcsel_count in range(vcsel_num):
                plane = gaus_list[vcsel_count]
                plane = gen_propTF_in_material(plane, frame_size, dz_micro_lens * frames_count, 1.53)
                plane_total += (abs(plane) ** 2)
            arr_total.append(abs(plane_total.numpy()))
            plane_total = torch.zeros((res, res), dtype=torch.cfloat)
            frames_count += 1

        # micro-micro_lens_and_big_lens_superposition
        micro_lens_idx_start = len(arr_total)
        for _ in range(plane_delay):
            arr_total.append(func_microlens_and_big_lens.numpy())


        print("Movie - done with superposition lens...")

        with torch.no_grad():
            frames_count = 0
            while (dz_big_lens * frames_count) < f_big_lens:
                plane_total = torch.zeros((res, res), dtype=torch.cfloat)
                for vcsel_count in range(vcsel_num):
                    plane = gaus_list[vcsel_count]
                    plane = gen_propTF_in_material(plane, frame_size, f_micro_lens, photo_polymer)
                    plane = plane * phase_microlens_and_big_lens
                    plane = gen_propTF_in_material(plane, frame_size, dz_big_lens * frames_count, air)
                    plane_total += abs(plane) ** 2
                arr_total.append(abs(plane_total.numpy()))
                frames_count += 1
        print("done movie only optics")

    return arr_total, micro_lens_idx_start