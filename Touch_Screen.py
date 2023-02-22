#adb -s AEB00078158 logcat -s "MyLog"
#('adb.exe -s AEB00078158 logcat -s "MyLog"'.split())
#adb -s AEB00078158 logcat -s "MyLog" | python C:\Temp\Android_Logcat_reading.py
#adb -s AEB00078158 logcat -s "MyLog" | python "L:\Laser to Tablet\Touch_Screen.py"
import mouse, keyboard
import _thread
import subprocess
import os
import queue as Queue
import subprocess
import threading
import datetime
import time

from win32api import GetSystemMetrics


def get_coordinate():
        try:
            try:
                    
                x = input().split()[-2]
                y = input().split()[-1]
                #print('x: %s y: %s' % (int(x), int(y)))
                return float(x), float(y)
                
            except ValueError:
                return -1, -1
        
        except EOFError:
            # no more information
                return -2, -2
        

def Resolution_Calibration ():
        res_x = 0
        res_y = 0
        print ('\nCalibrating resolution\n for 5 seconds touch the right most bottom pixel on your screen multiple time to get your phone"s screen resolution\n')
        start = time.time()
        while (time.time() - start < 5):
                x, y = get_coordinate()
                res_x = max (res_x, x)
                res_y = max (res_y, y)
        print ("Phone screen resolution: ",res_x," x ",res_y)
        return float(res_x), float(res_y)

def rotate_x_y (x, y, res_x, res_y, rotate_option):
        if (rotate_option == 0):
                return x/res_x, y/res_y
        if (rotate_option == 180):
                return (res_x-x)/res_x, (res_y - y)/res_y
        if (rotate_option == 90):
                return (res_y - y)/res_y, x/res_x
        if (rotate_option == 270):
                return y/res_y, (res_x - x)/res_x

def scale_x_y (x, y, scale_option):
         x = (x-0.5) * (1/scale_option) + 0.5
         y = (y-0.5) * (1/scale_option) + 0.5
         return x, y
#0 90 180 270
rotate_option = 90
#value (0.....1]
scale_option = 0.5
x = 0
y = 0
#res_x = 0;
#res_y = 0;
res_x, res_y = Resolution_Calibration();

res_width = GetSystemMetrics(0)
res_height = GetSystemMetrics(1)
print ("Pc: ",res_width," x ",res_height)
print ("Rotated: ", rotate_option, " degree clock-wise")
print ("Begin, press ctrl + c inside this cmd window to stop")

while (not (x == -2)):
        x, y = get_coordinate()
        x, y = rotate_x_y (x, y, res_x, res_y, rotate_option)
        x, y = scale_x_y (x, y, scale_option)
        mouse.move (x*res_width, y*res_height)
