#!/usr/local/bin/python3
# Simple program, captures the specified screenshot in a particular
# region.
import argparse
import logging
import torch
import time
import pyautogui
from model import Net
from torchvision import transforms
# mss is significantly faster than using screencapture.
import mss
import mss.tools
from PIL import Image


SCREENSHOT_DIR = '/tmp/screenshots/'
START_DELAY = 1
# Entire Game Region at lowest resolution.
ENTIRE_GAME_REGION = (0, 50, 750, 450)

# Game region training data is based upon.
EXPECTED_REGION = (135, 170, 440, 365)
EXPECTED_REGION = {'top': 120, 'left': 135, 'width': 440, 'height': 365}


def outputMouseLocations():
    """
    Used for debugging by outputting current positions of the mouse
    """
    while True:
        x, y = pyautogui.position()
        print('X: {},Y: {}'.format(x, y))
        time.sleep(0.2)


class Player:
    def __init__(self, fps, model_file):
        self.fps = fps
        # Set up the model
        self.model = Net()
        self.model.load_state_dict(torch.load(args.model_file))
        self.model.eval()

        self.sct = mss.mss()
        # Stores the previous key pressed down
        self.prev_key = ''
        self.loader = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # Stores the game status of past games.
        self.history = []
        self.current_status = None

    def capture_screenshot(self, region, filename):
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
        # Grab the data
        img = self.sct.grab(region)
        # Transform to PIL to pass to the net
        pil_img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        if filename:
            # Save to the picture file
            mss.tools.to_png(img.rgb, img.size, output=filename)
        return pil_img

    def gameover(self):
        # Test whether the top-right pixel is black (game going)
        img = self.sct.grab({'top': 70, 'left': 765, 'width': 1, 'height': 1})
        return img.pixels[0][0] != (0, 0, 0)

    def execute_key_press(self, key, num_times):
        """
        Presses one of the supported keys (l,r,s) corresponding to left,right,
        None num_times.
        """
        supported_keys = {0: 'left', 1: 'right', 2: ''}

        if key not in supported_keys:
            logging.error('Unexpected key passed:', key)

        key_to_press = supported_keys[key]
        if key_to_press:
            pyautogui.press([key_to_press] * num_times)
        logging.info('TIME[{}]Pressing {} Num Times: {}'.format(time.time(),
                                                                key_to_press,
                                                                num_times))

    def execute_key(self, key):
        """
        Holds down one of the keys to move (releasing the previous key if it's
        different).
        """
        supported_keys = {0: 'left', 1: 'right', 2: ''}
        if key not in supported_keys:
            logging.error('Unexpected key passed:', key)

        key_to_press = supported_keys[key]
        if self.prev_key and self.prev_key != key_to_press:
            pyautogui.keyUp(self.prev_key)
        if key_to_press:
            pyautogui.keyDown(key_to_press)
        logging.info('TIME[{}] Moving {} '.format(time.time(), key_to_press))
        self.prev_key = key_to_press

    def start_game(self):
        logging.info('\nStarting the Game\n')
        pyautogui.moveTo(200, 200, duration=1)
        pyautogui.click(clicks=2, interval=1)
        pyautogui.press(['space'] * 2)

    def _image_loader(self, image):
        """
        Use to load the image into a tensor using our loader.
        """
        # TODO(kbaichoo): support cuda.
        image = self.loader(image).float()
        return image
        # return image.cuda()

    def clear_game(self):
        """
        Resets the agent to the state prior it payed the game.
        Stores the game results in the history.
        """
        if self.prev_key != '':
            pyautogui.keyUp(self.prev_key)
        # Store the game status
        if self.current_status:
            self.history.append(self.current_status)
        self.current_status = None

    def play_game(self):
        """
        Plays a single game of SuperHexagon.
        """
        image_count = 0
        start_time = time.time()
        stats = GameStats()
        self.start_game()
        # TODO(kbaichoo): use a real rate limiter based on how long it takes
        # to process the file, etc..
        while True:
            image_count += 1

            # Test whether the game is over.
            if image_count % 50 == 0 and self.gameover():
                break
            image_name = SCREENSHOT_DIR + str(image_count) + '.png'
            image = self.capture_screenshot(EXPECTED_REGION, image_name)
            image = self._image_loader(image)
            image.unsqueeze_(0)

            output = self.model(image)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            stats.predictions[pred.item()] += 1
            self.execute_key(pred.item())
        # Save the stats for this run
        end_time = time.time()
        stats.game_time = end_time - start_time
        self.current_status = stats
        logging.info('Game ended after %d seconds', stats.game_time)

    def play_games(self, n):
        """
        Plays n games, returning all stats for the games played.
        """
        self.clear_game()
        self.history = []

        for i in range(n):
            logging.info('\nPlaying Game {} of {}\n'.format(i, n))
            self.play_game()
            self.clear_game()
        return self.history


class GameStats:
    def __init__(self):
        self.game_time = 0
        # Keep an array of num times we predicted that value
        self.predictions = [0] * 3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO(kbaichoo): add arguments for directory to output images to,
    # path for model, region location, etc..
    parser.add_argument('--fps',
                        help='FPS between screenshots', type=int,
                        required=True)
    parser.add_argument('--model_file',
                        help='path to the model', required=True)

    args = parser.parse_args()
    fps = args.fps

    if fps >= 8:
        logging.warn('Cannot guarentee >= 8 fps screenshots...')

    logging.getLogger().setLevel(logging.INFO)

    print('Launched main.py...')
    print('Waiting for {} seconds before beginning'.format(START_DELAY))
    if SCREENSHOT_DIR:
        print('screenshots to directory: {}'.format(SCREENSHOT_DIR))
    time.sleep(START_DELAY)

    # Between each call to pyautogui in seconds
    pyautogui.PAUSE = 0.0

    player = Player(args.fps, args.model_file)
    stats = player.play_games(10)
    for stat in stats:
        print('left | right | stay')
        print(stat.predictions)
