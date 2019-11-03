#!/usr/local/bin/python3
# A simple example going to a text window and typing hello.
# at first it misses the h, and then goes back to type it.
# It seems like clicking in OS X Mojave isn't working well...
# So clicks don't change the window in focus.
# Keys are fine though, which are all we need.
import time
import pyautogui
screenWidth, screenHeight = pyautogui.size()
currentMouseX, currentMouseY = pyautogui.position()
print('Screen Height:{}, Width:{}'.format(screenHeight, screenWidth))
print('Current Pos: X:{}, Y{}'.format(currentMouseX, currentMouseY))

pyautogui.moveTo(200, 200, duration=1)
# Left-clicks don't aren't clicking for me.
pyautogui.click(clicks=2, interval=2)

print('Sleeping... focus the window to type on.')
time.sleep(10)

# Enter insert mode,
pyautogui.press(['e','l','l','o'])
pyautogui.press(['left'] * 4)
pyautogui.press(['h'])
print('Done typing.')


