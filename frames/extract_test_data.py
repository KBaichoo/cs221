#!/usr/local/bin/python3
# this takes all the image files in "left", "right", and "stay"
# in "train" and moves them over to "test"
import os
import shutil


def move_file(source_file, directory):
    moved_path = './test/' + directory + '/' + source_file
    shutil.move('./train/' + directory + '/' + source_file, moved_path)


if __name__ == '__main__':
    relevant_dirs = ["left", "right", "stay"]
    for type_dir in relevant_dirs:
        for root, dirs, files in os.walk("./train/" + type_dir):
            filtered_files = list(filter(lambda x: x.endswith('.jpg'), files))
            count = 1
            for image_file in filtered_files:
                if count == 10:
                    move_file(image_file, type_dir)
                    count = 1
                else:
                    count = count + 1
