import os

dir = "/media/vmc/Data/VMC/Workspace/IS-Car/Images"

def load_image(dir):
    image_path = []
    labels = []

    labels_name = os.listdir(dir)
    for label in labels_name:
        label_path = os.path.join(dir, label)
        for image_filename in os.listdir(label_path):
            if(os.path.splitext(image_filename)[1] != '.csv'):
                image_path.append(os.path.join(label_path, image_filename))
                labels.append(label)

    return image_path, labels


load_image(dir)