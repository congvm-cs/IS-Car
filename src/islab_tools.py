import os
import cv2
from skimage.feature import hog


def load_image(dataset_dir):
    print("Loading Image")
    image_path = []
    labels = []

    labels_name = os.listdir(dataset_dir)
    for label in labels_name:
        label_path = os.path.join(dataset_dir, label)
        for image_filename in os.listdir(label_path):
            if os.path.splitext(image_filename)[1] != '.csv':
                image_path.append(os.path.join(label_path, image_filename))
                labels.append(label)
    return image_path, labels


def hog_compute(images):
    print("Compute HOG")
    image_arr = []
    for image in images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
        image_arr.append(hog_image)
    return image_arr

