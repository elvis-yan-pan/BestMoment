#!/usr/bin/env python
import cv2
import os
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
    return img_folder #delete_last(input_path) + "data"


if __name__ == "__main__":
    video_cut("videos/1581737285710490.mp4")
