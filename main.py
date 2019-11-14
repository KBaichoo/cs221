#!/usr/local/bin/python3
# Simple program, captures the specified screenshot in a particular
# region.
import argparse
import logging
import torch
import time
import pyautogui
import subprocess
import os
from model import Net


SCREENSHOT_DIR = '/tmp/screenshots/'
START_DELAY = 5
EXPECTED_REGION = (0,50,750,450)


def captureScreenshot(region, filename):
    """
    Calls captureScreenshot outputting it to a file if specified, otherwise
    it's in memory.

    Screenshot takes ~100mS, so can't get sub-granularity of this.

    Args:
        region - a 4 tuple (left, top, width, height)
        filename - optional filename for the image if want to save it
    Returns:
        The image object.
    """
    if filename:
        return pyautogui.screenshot(filename,region=region)
    else:
        return pyautogui.screenshot(region=region)

def outputMouseLocations():
    """
    Used for debugging by outputting current positions of the mouse
    """
    while True:
        x, y = pyautogui.position()
        print('X: {},Y: {}'.format(x,y))
        time.sleep(0.2)

def ExecuteKey(key, num_times):
    """
    Presses one of the supported keys (l,r,s) corresponding to left,right,None
    num_times.
    """
    supported_keys = {'l': 'left', 'r': 'right', 's': ''}

    if key not in supported_keys:
        logging.error('Unexpected key passed:', key)

    key_to_press = supported_keys[key]

    if key_to_press:
        for i in range(0, num_times):
            pyautogui.press(key_to_press)
    logging.info('Pressing {} Num Times: {}'.format(key_to_press, num_times))

def StartGame():
    pyautogui.moveTo(200, 200, duration = 1)
    pyautogui.click(clicks=2, interval=1)
    pyautogui.press(['space'] * 2)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO(kbaichoo): add arguments for directory to output images to,
    # path for model, region location, etc..
    parser.add_argument('--fps',
            help='FPS between screenshots', type=int, required=True)
    parser.add_argument('--model_file',
            help='path to the model', required=True)

    args = parser.parse_args()
    fps = args.fps

    if fps >= 8:
        logging.warn('Cannot guarentee >= 8 fps screenshots...')

    logging.getLogger().setLevel(logging.INFO)

    # Set up the model
    model = Net()
    model.load_state_dict(torch.load(args.model_file))
    model.eval()

    print('Launched grab_screenshot.py...')
    print('Waiting for {} seconds before beginning'.format(START_DELAY))
    if SCREENSHOT_DIR:
        print('screenshots to directory: {}'.format(SCREENSHOT_DIR))
    time.sleep(START_DELAY)

    print('Starting')
    StartGame()
    
    image_count = 0
    while True:
        image_count += 1
        image_name = SCREENSHOT_DIR + str(image_count) + '.png'
        image = captureScreenshot(EXPECTED_REGION, image_name)

        # TODO(kbaichoo): call to NN

        key, num_times = ('l', 10)
        ExecuteKey(key, num_times)

        # TODO(kbaichoo): use a real rate limiter based on how long it takes
        # to process the file, etc..
        time.sleep(1.0/fps)
