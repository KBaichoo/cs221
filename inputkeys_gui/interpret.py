#!/usr/local/bin/python3
# Simple program that runs tail in a subprocess on some file
# It then inteprets that input using pygui to interact with
# our target app.
import time
import pyautogui
import subprocess

# Home to application
pyautogui.moveTo(200, 200, duration=1)

pyautogui.click(clicks=2, interval=1)

print('Sleeping... focus the window to type on.')
time.sleep(10)

# Launch tail -f on a file.
filename = "/tmp/game_input"
f = subprocess.Popen(['tail','-F',filename],\
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)

# Decode tailed lines as they come in
while True:
    line = f.stdout.readline()
    # Interpret lines read to do something with pyautogui.
    if 'left' in line.decode('utf-8'):
        pyautogui.press(['l'])

