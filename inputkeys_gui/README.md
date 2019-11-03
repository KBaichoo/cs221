See https://github.com/asweigart/pyautogui
instaillation in order to use the program.

Note:
Clicking functionality for OS X broke in:
'pyautogui>=0.9.45'

So do a:
`pip3 install 'pyautogui==0.9.44'`

More details:
https://github.com/asweigart/pyautogui/issues/369

Attached are an example script using this functionality to write into the 
textbox.

There's also an example with `./5_left.sh & ./interpret.py` that is 
essentially the backbone of our application. The left scripts writes
output to some tmp file, that interpret reads and uses pygui to interact
with the application. We just need to plug in our NN model to write to
the tmp file.

For OS X:
See permissions to control computer:
https://github.com/asweigart/pyautogui/issues/247
