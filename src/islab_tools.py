import os
import cv2
from skimage.feature import hog
import numpy as np

def load_data_from_txt(dataset_dir):
    print("Loading data from txt...")
    data = []
    labels = []
    data_temp = []

    labels_name = os.listdir(dataset_dir)
    for label in labels_name:
        label_path = os.path.join(dataset_dir, label)
        for txt_filename in os.listdir(label_path):
            if os.path.splitext(txt_filename)[1] == '.txt':
                txt_path = os.path.join(label_path, txt_filename)

                # READ ALL DATA FROM TXT
                data_temp.clear()
                with open(txt_path, 'r') as file:
                    for line in file:
                        data_temp.extend(line.splitlines())

                data.append(data_temp)
                labels.append(label)
    return data, labels

def load_images_and_labels(dataset_dir, new_size = [64, 64]):
    print("Loading Image...")
    labels = []
    images = []
    labels_name = os.listdir(dataset_dir)
    for label in labels_name:
        label_path = os.path.join(dataset_dir, label)
        for image_filename in os.listdir(label_path):
            if os.path.splitext(image_filename)[1] != '.csv':
                img = cv2.imread(os.path.join(label_path, image_filename))
                img = cv2.resize(img, (new_size[0], new_size[1]))
                images.append(img)
                labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def hog_compute(images):
    print("Compute HOG...")
    image_arr = []
    for image in images:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualise=True)
        image_arr.append(hog_image)
    return image_arr

