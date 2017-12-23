"""
===============================
Histogram of Oriented Gradients
===============================

Algorithm overview
------------------

Compute a Histogram of Oriented Gradients (HOG) by

1. (optional) global image normalisation
2. computing the gradient image in x and y
3. computing gradient histograms
4. normalising across blocks
5. flattening into a feature vector
"""

from src.islab_tools import load_images_and_labels
from src.islab_tools import load_data_from_txt
import pickle

# dir = "__________Image directory____________"
dir = "/media/vmc/Data/VMC/Workspace/IS-Car/Images"
txt_path = '/media/vmc/Data/VMC/Workspace/IS-Car/data/'


def main():
    # images, labels = load_image(dir)
    # image_arr = hog_compute(images)
    # pickle.dump([image_arr, labels], open("/models/data_set.sav", "wb"))
    #
    # with open("./models/data_set.sav", "rb") as file:
    #     x, y = pickle.load(file)
    #
    # print(x)

    data, labels = load_data_from_txt(txt_path)
    print(data)
    print('done!')


if __name__ == "__main__":
    main()