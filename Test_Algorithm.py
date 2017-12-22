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

from src.islab_tools import load_image
from src.islab_tools import hog_compute
import pickle

# dir = "__________Image directory____________"
dir = "/media/vmc/Data/VMC/Workspace/IS-Car/Images"

def main():
    images, labels = load_image(dir)
    image_arr = hog_compute(images)
    pickle.dump([image_arr, labels], open("/models/data_set.sav", "wb"))

    with open("./models/data_set.sav", "rb") as file:
        x, y = pickle.load(file)

    print(x)


if __name__ == "__main__":
    main()