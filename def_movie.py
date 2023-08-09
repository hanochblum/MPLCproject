import os
import shutil
import cv2
from matplotlib import pyplot as plt

from create_directory import create_directory
from def_param import res, fps ,plane_delay
from movie_MPLC import gen_movie_MPLC_intensity, gen_movie_MPLC_intensity_no_phase_masks
from def_plane import gaus_list


def def_movie(input, weights, movie_type):

    if movie_type == "MPLC":
        arr, micro_lens_idx_start = gen_movie_MPLC_intensity(gaus_list, weights)
        video_name = "MPLC.avi"
        image_directory_name = "ALL_VCSELs_images_for_" + video_name
        create_directory(image_directory_name)

    if movie_type == "only_lens":
        arr, micro_lens_idx_start = gen_movie_MPLC_intensity_no_phase_masks(input, gaus_list)
        video_name = "only lens.avi"
        image_directory_name = "images_for_" + video_name
        create_directory(image_directory_name)



    # put images to the new directory
    for image_num in range(len(arr)):
        images_path = 'image/' + image_directory_name + '/img'
        img_name = str(image_num) + ".png"
        if image_num < micro_lens_idx_start or image_num > (micro_lens_idx_start + plane_delay-1):
            plt.imsave(images_path + img_name, arr[image_num])
        else:
            plt.imsave(images_path + img_name, arr[image_num], cmap='bone')

    # sorting images to list
    image_folder = r"image/" + image_directory_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # create video
    video_folder_path = "video/"
    video = cv2.VideoWriter(video_folder_path + video_name, 0, fps, (res, res))
    cur_image_folder = r"image/" + image_directory_name + "/img"
    for ii in range(len(arr)):
        cur_image_folder2 = cur_image_folder + str(ii) + ".png"
        video.write(cv2.imread(cur_image_folder2))


    # delet new images directory
    images_file_path = r"image/" + image_directory_name
    if os.path.exists(images_file_path):
        try:
            shutil.rmtree(images_file_path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))


    cv2.destroyAllWindows()
    video.release()

    return