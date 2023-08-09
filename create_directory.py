import os
import shutil


def create_directory(image_directory_name):

    images_file_path = r"image/" + image_directory_name
    if os.path.exists(images_file_path):  # delete image folder from previous run.
        try:
            shutil.rmtree(images_file_path)
            print("Image folder deleted successfully")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    images_path = r"image"
    path = os.path.join(images_path, image_directory_name)
    os.mkdir(path)