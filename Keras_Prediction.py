import cv2
import numpy as np
from keras.models import load_model
model = load_model('D:\\model\\model_traffic_signs.h5py')
#print(model.summary())

img = cv2.imread('C:\\Users\\Win 8.1 VS8 X64\\Desktop\\123.jpg')
x = cv2.resize(img,(28, 28))
print(x.shape)
x = x/255
x = x.reshape(1, 28, 28, 3)

#print(model.predict(x))
print(np.argmax(model.predict(x)))

''''''
# img2 = cv2.imread('C:\\Users\\Win 8.1 VS8 X64\\Desktop\\Data2\\1\\00000_00000.ppm')
# cv2.imshow('stop',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()