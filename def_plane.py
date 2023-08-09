
import torch
from big_lens import gen_big_lens
from def_param import res, M_square, vcsel_num
from fiber_mode import gen_fiber_mode
from gen_VCSELs import gen_vcsel
from micro_lens import gen_micro_lens


gaus_total, gaus_list, max_list = gen_vcsel()


phase_plane_micro_lens, func_plane_micro_lens, delta_x2, plane_func_micro_lens_structure = gen_micro_lens(max_list)
plane_phase_micro_lens_structure = torch.exp(1j*plane_func_micro_lens_structure)

phase_big_lens, func_big_lens, delta_x = gen_big_lens()

# make from both lens -> one lens
func_microlens_and_big_lens = plane_func_micro_lens_structure + func_big_lens
phase_microlens_and_big_lens = plane_phase_micro_lens_structure * phase_big_lens
modes_to_calc = int(M_square)
fiber_mode_list = torch.zeros((modes_to_calc, res, res), dtype=torch.float64, requires_grad=True)
plane_area_list = torch.zeros((modes_to_calc, res, res), dtype=torch.cfloat)
gen_fiber_in_center = True
all_fiber_modes_intensity = torch.zeros((res, res), dtype=torch.cfloat)
vec_new_mode_eff = torch.zeros(1, modes_to_calc)
with torch.no_grad():
    for mode_cur in range(modes_to_calc):
        fiber_mode_list[mode_cur], plane_area_list[mode_cur] = gen_fiber_mode(mode_cur+1)
        fiber_mode_cur = fiber_mode_list[mode_cur]
        all_fiber_modes_intensity += abs(fiber_mode_cur)**2
        print("calculate mode number " + str(mode_cur+1))


coupling_matrix_with_phase_mask = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.float64)
coupling_matrix_overlapintegral_with_phase_mask = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.cfloat)
coupling_matrix_only_optics = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.float64)
coupling_matrix_overlapintegral_only_optics = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.cfloat)
coupling_matrix_eff = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.float64)
coupling_matrix_overlap = torch.zeros((vcsel_num, modes_to_calc), dtype=torch.cfloat)
