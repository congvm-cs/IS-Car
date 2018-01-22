import cv2
import time
import numpy as np
from keras.models import load_model
model = load_model('D:\\Research\\Autonomous Car\\Models\\40Sign_Model.h5py')

img_sample = cv2.imread('40a.jpg')
img1 = cv2.resize(img_sample,(640, 480))

m = img1.shape[0]
n = img1.shape[1]

m = int(m)
n = int(n)

print(m)
print(n)

img2 = cv2.resize(img1, (int(n*3/4), int(m*3/4)))
img3 = cv2.resize(img1, (int(n/2), int(m/2)))

x=0
y=0

WinH = 100
WinW = 100
Stride = 50

print(img1.shape[0])
print(img1.shape[1])

# i=1
max = 0.5

# img = img1.copy()
# while 1:
# if i == 1:
#     img = img1.copy()
# if i == 2:
#     img = img2.copy()
# if i == 3:
#     img = img3.copy()
for i in range(3):
    for y in range(0, img1.shape[0]-3*WinW, Stride):
        for x in range(0, img1.shape[1]-3*WinH, Stride):
            # img = img1.copy()
            if i == 0:
                img = img1.copy()
            if i == 1:
                img = img2.copy()
            if i == 2:
                img = img3.copy()

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

            if stop1[0] > max:
                max = stop1[0]
                # print(x, y, x + WinW, y + WinH)
                q = y
                w = x
                print(max)

            #prediction = np.argmax(model.predict(x))
            time.sleep(0.001)
            cv2.waitKey(1)
    i = i + 1

print(max)
print(w, q, w + WinW, q + WinH)
crop_img1 = img[q:q + WinH, w:w + WinW]
cv2.imshow("cropped1", crop_img1)
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