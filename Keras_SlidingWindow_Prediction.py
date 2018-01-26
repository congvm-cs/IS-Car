import cv2
import time
import numpy as np
from keras.models import load_model
model = load_model('D:\\Research\\Autonomous Car\\Models\\model.h5')

img_sample = cv2.imread('C:\\Users\\Win 8.1 VS8 X64\\Desktop\\stopb.jpg')
img1 = cv2.resize(img_sample,(640, 480))

# m = img1.shape[0]
# n = img1.shape[1]
#
# m = int(m)
# n = int(n)
#
# print(m)
# print(n)
#
# img2 = cv2.resize(img1, (int(n*3/4), int(m*3/4)))
# img3 = cv2.resize(img1, (int(n/2), int(m/2)))

x=0
y=0

WinH = 0
WinW = 0
Stride = 50

print(img1.shape[0])
print(img1.shape[1])

# i=1
max = 0
img = img1.copy()

for i in range(4):
    # if i == 0:
    #     WinH = 250
    #     WinW = 250
    if i == 0:
        WinH = 200
        WinW = 200
    if i == 1:
        WinH = 150
        WinW = 150
    if i == 2:
        WinH = 100
        WinW = 100
    if i == 3:
        WinH = 50
        WinW = 50
    for y in range(0, img.shape[0]-WinW, Stride):
        for x in range(0, img.shape[1]-WinH, Stride):
            img = img1.copy()
            imgr = cv2.rectangle(img, (x, y), (x + WinH, y + WinW), (0, 255, 0), 2)
            cv2.imshow('image', imgr)

            crop_img = img[y:y + WinH, x:x + WinW]
            cv2.imshow("cropped", crop_img)
            xx = cv2.resize(crop_img, (32, 32))
            #print(x.shape)
            xx = xx / 255
            xx = xx.reshape(1, 32, 32, 3)

            stop = model.predict(xx)
            stop1 = stop[0]
            # print(stop1[0])

            if stop1[1] > max:
                max = stop1[1]
                # print(x, y, x + WinW, y + WinH)
                q = y
                w = x
                size_Win = WinH
                print(max)

            #prediction = np.argmax(model.predict(x))
            time.sleep(0.001)
            cv2.waitKey(1)
    i = i + 1

print(max)
print(w, q, w + size_Win, q + size_Win)
crop_img1 = img[q:q + size_Win, w:w + size_Win]
cv2.imshow("cropped1", crop_img1)

bounding = cv2.rectangle(img1, (w, q), (w + size_Win, q + size_Win), (0, 255, 0), 2)
cv2.imshow('bounding box', bounding)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
x = cv2.resize(img,(32, 32))
print(x.shape)
x = x/255
x = x.reshape(1, 32, 32, 3)

#print(model.predict(x))
print(np.argmax(model.predict(x)))

# img2 = cv2.imread('C:\\Users\\Win 8.1 VS8 X64\\Desktop\\Data2\\1\\00000_00000.ppm')
# cv2.imshow('stop',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
'''