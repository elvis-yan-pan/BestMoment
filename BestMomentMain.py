#!/usr/bin/env python
import face_recognition
from PIL import Image
from itertools import chain
import matplotlib.pyplot as plt
import os
import re
import multiprocessing
import itertools
import sys
import numpy as np
import click
import cv2
import shutil


def delete_last(s):
    revs = s[::-1]
    for char in revs:
        if char == "/":
            break
        revs = revs[1:]
    return revs[::-1]

def video_cut(input_path, img_folder):
    # delete the images of the data folder
    # img_folder = './data/'
    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # read video file
    cam = cv2.VideoCapture(input_path)
    try:
        if not os.path.exists("data"):
            os.makedirs("data")
    except OSError:
        print("Error")
    count = 0
    while True:
        ret, frame = cam.read()
        if ret:
            if count % 5 == 0:
                name = './data/frame' + str(int(count / 5)).zfill(3) + ".jpg"
                cv2.imwrite(name, frame)
            count += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
    return delete_last(input_path) + "data"


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def load_image(file):
    img = face_recognition.load_image_file(file)
    return img


def visualize_landmark(img_array):
    face_landmarks_list = face_recognition.face_landmarks(img_array)
    list_values = []
    for p in face_landmarks_list:
        list_values.extend([i for i in p.values()])
    [x, y] = list(zip(*list(chain.from_iterable(list_values))))
    img = Image.fromarray(img_array)
    plt.imshow(img)
    plt.scatter(x, y, c="r", s=5)
    plt.show()


def find_faces(image, output_path):
    face_locations_list = face_recognition.face_locations(image)
    for face in face_locations_list:
        (t, r, b, l) = face
        img = Image.fromarray(image[t:b, l:r, :])
        img.save(output_path)
        # plt.imshow(img)
        # plt.show()


def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


def print_result(file_name, name, distance, show_distance=False):
    if show_distance:
        print("{},{},{}".format(file_name, name, distance))
    else:
        print("{},{}".format(file_name, name))


def test_image(image_to_check, known_names, known_face_encodings, tolerance=0.6, show_distance=False):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), Image.LANCZOS)
        unknown_image = np.array(pil_img)
    find_faces(unknown_image, image_to_check.replace("frame", "you"))

    unknown_encodings = face_recognition.face_encodings(unknown_image)

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        # print_result(image_to_check, "no_persons_found", None, show_distance)
        return [[], [], []]
    else:
        return [result, known_names, list(distances)]


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )
    # print("Hello")
    res = pool.starmap(test_image, function_parameters)
    return res


def compare_faces(known_people_folder, image_to_check, cpus, tolerance, show_distance):
    known_names, known_face_encodings = scan_known_people(known_people_folder)

    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(image_to_check):
        if cpus == 1:
            image_files = image_files_in_folder(image_to_check)
            image_files.sort()
            res = [test_image(image_file, known_names, known_face_encodings, tolerance, show_distance) for image_file in image_files]
        else:
            res = process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
    else:
        res = [test_image(image_to_check, known_names, known_face_encodings, tolerance, show_distance)]

    return res  # res : (result, known_names, distances) list


def find_best_fit(res, known_people_folder, image_folder):
    known_people = image_files_in_folder(known_people_folder)
    best_fits = []
    min_vals = []
    ext = ['.jpeg', '.jpg', '.png']
    for i in range(len(res)):
        p = res[i]
        if p[2] != []:
            min_vals.append(min(p[2]))
            min_index = p[2].index(min(p[2]))
            known_img = p[1][min_index]
            best_fits.append(known_img)
        else:
            min_vals.append(1)
            best_fits.append("not_found")

    if min(min_vals) == 1:
        print("Not found")
        return False
    best_fit_index = min_vals.index(min(min_vals))
    best_pic = os.path.join(image_folder, "you{}.jpg".format(str(best_fit_index).zfill(3)))
    best_folder = os.path.join(image_folder, "best")
    os.mkdir(best_folder)
    shutil.copy(best_pic, os.path.join(image_folder, "best/you.jpg"))
    for j in ext:
        image_name = os.path.join(known_people_folder, (known_img + j))
        if os.path.exists(image_name):
            link = os.path.join(image_folder, "best/known.jpg")
            shutil.copy(image_name, link)
    return True
    # for i in range(len(res)):
    #     for j in ext:
    #         image_name = os.path.join(known_people_folder, (known_img + j))
    #         print(image_name)
    #         if os.path.exists(image_name):
    #             link = os.path.join(image_folder, "known_{}.jpg".format(str(i)))
    #             shutil.copy(image_name, link)


# @click.command()
# @click.argument('known_people_folder', default="./known_images")
# @click.argument('image_to_check', default="images/PY.jpeg")
# @click.argument('video', default="videos/1581792442143627.mp4")
# @click.argument('image_folder', default="./data/")
# @click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
# @click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
# @click.option('--show-distance', default=True, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
def main(known_people_folder, video, image_folder, cpus, tolerance, show_distance):
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    video_cut(video, image_folder)
    res = compare_faces(known_people_folder, image_folder, cpus, tolerance, show_distance)
    b = find_best_fit(res, known_people_folder, image_folder)
    if b:
        return "Great! Your images are in /data. Check it out!"
    else:
        return "We can't find you! Select a better video!"
