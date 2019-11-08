#!/usr/local/bin/python3
# Note you need to have made the "left", "right" and "stay" folders prior to
# running, and have installed PIL.
from PIL import Image
import os
import shutil


def move_file(source_file, directory):
    moved_path = './' + directory + '/' + source_file
    shutil.move(source_file, moved_path)



if __name__ == '__main__':
    print('To classify use "a" for left, "s" for stay and "d" for right')
    for root, dirs, files in os.walk("."):
        print('Looking at current directory...')
        filtered_files = list(filter(lambda x: x.endswith('.jpg'), files))
        for image_file in filtered_files:
            with Image.open(image_file) as img:
                print('Showing image file {}'.format(image_file))
                img.show()
                while True:
                    classification = input('Enter classification:')

                    if classification == 'a':
                        # Move left
                        move_file(image_file, 'left')
                        break
                    elif classification == 's':
                        # Move stay
                        move_file(image_file, 'stay')
                        break
                    elif classification == 'd':
                        # Move right
                        move_file(image_file, 'right')
                        break
                    else:
                        print('To classify use "a" for left, "s" for stay '
                                'and "d" for right')

            
