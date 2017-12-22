import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import time
import pyramid
import argparse

img1 = cv2.imread('blue.png')

img2 = cv2.resize(img1, ((img1.shape[1]/2), (img1.shape[0]/2)))
x=0
y=0

WinH = 128
WinW = 128
Stride = 64

print img1.shape[0]
print img1.shape[1]

i=1

while 1:
    if i == 1:
        img = img1.copy()
    else:
        img = img2.copy()

    for y in range(0, img.shape[0]-WinW, Stride):
        for x in range(0, img.shape[1]-WinH, Stride):
            if i == 1:
                img = img1.copy()
            else:
                img = img2.copy()
            img = cv2.rectangle(img, (x, y), (x + WinH, y + WinW), (0, 255, 0), 2)
            cv2.imshow('image', img)
            time.sleep(0.001)
            cv2.waitKey(1)
    i=~i