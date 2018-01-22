# -*- coding: utf-8 -*-
# %matplotlib inline
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from skimage.feature import hog
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### TRAINING HOG
# Parameters used during feature extraction
params = {}
params['color_space'] = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
params['orient'] = 9              # HOG orientations
params['pix_per_cell'] = 8        # HOG pixels per cell
params['cell_per_block'] = 2      # HOG cells per block
params['hog_channel'] = 'ALL'     # Can be 0, 1, 2, or "ALL"
params['spatial_size'] = (32, 32) # Spatial binning dimensions
params['hist_bins'] = 32          # Number of histogram bins
params['spatial_feat'] = True     # Spatial features on or off
params['hist_feat'] = True        # Histogram features on or off
params['hog_feat'] = True         # HOG features on or off
print(params)
print('-----------------------------------------------------------')

### LOADING DATA
def get_data(folder):
    print('load data...')

    TRAIN_FILE = "train.p"
    VALID_FILE = "valid.p"
    TEST_FILE = "test.p"

    # Load the dataset
    training_file = os.path.join(folder, TRAIN_FILE)
    validation_file= os.path.join(folder, VALID_FILE)
    testing_file =  os.path.join(folder, TEST_FILE)

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Retrive all datas
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


def extract_features(X_train, params):            
    color_space    = params['color_space']
    spatial_size   = params['spatial_size']
    hist_bins      = params['hist_bins']
    orient         = params['orient']
    pix_per_cell   = params['pix_per_cell']
    cell_per_block = params['cell_per_block']
    hog_channel    = params['hog_channel']
    spatial_feat   = params['spatial_feat']
    hist_feat      = params['hist_feat']
    hog_feat       = params['hog_feat']
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for every_X_train in X_train: 
        img_features = single_img_features(every_X_train, color_space=color_space, spatial_size=spatial_size,
                        hist_bins=hist_bins, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                        spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
    
        features.append(img_features)
        
    # Return list of feature vectors
    return features

# with open('dataHOG.pkl', 'wb') as file:
#   pickle.dump([X_train_feats, y_train], file)


### TRAINING SVM
def load_dataset():
    TRAFFIC_SIGNS_PATH = 'C:/Users/VMC/Desktop/OwnCollection/traffic-signs'
    NOT_TRAFFIC_SIGNS_PATH = 'C:/Users/VMC/Desktop/OwnCollection/not-traffic-signs'

    # LOAD TRAFFIC SIGNS
    # X_ts, _, _, _, _, _ = get_data(TRAFFIC_SIGNS_PATH)
    
    X_ts, Y_ts = load_images_and_labels('C:\\Users\\VMC\\Desktop\\data', [32, 32])
    
    # LOAD NOT TRAFFIC SIGNS
    X_not_ts = []
    for _, folder_name in enumerate(os.listdir(NOT_TRAFFIC_SIGNS_PATH)):
        image_folder_path = os.path.join(NOT_TRAFFIC_SIGNS_PATH, folder_name)
        for i, file_name in enumerate(os.listdir(image_folder_path)):
            img = cv2.imread(os.path.join(image_folder_path, file_name))
            img = cv2.resize(img, (32, 32))
            X_not_ts.append(img)

    # Convert to array
    X_not_ts = np.array(X_not_ts)
    Y_not_ts = np.zeros(X_not_ts.shape[0])

    # CUT-OFF
    X_train = np.concatenate([X_ts, X_not_ts], axis=0)
    Y_train = np.concatenate([Y_ts, Y_not_ts], axis=0)

    print(X_train.shape)
    Y_train = np.array(Y_train)
    print(Y_train[2323:2356])

    print('length of X_ts: {}'.format(X_ts.shape))
    print('length of y_ts: {}'.format(Y_ts.shape))
    print('length of X_not_ts: {}'.format(X_not_ts.shape))
    print('length of Y_not_ts: {}'.format(Y_not_ts.shape))
    print('--------------------------------------------------------')
    print('length of X_train: {}'.format(X_train.shape))
    print('length of Y_train: {}'.format(Y_train.shape))
    print('--------------------------------------------------------')

    # Save dataset
    save_path = os.path.join(os.getcwd(), 'hog_traffic_dataset.pkl')
    with open('hog_traffic_dataset.pkl', 'wb') as file:
        pickle.dump([X_train, Y_train], file)
        print('save data file at {}'.format(save_path))

    # with open('hog_traffic_dataset.pkl', 'rb') as file:
    #     X_scaler, scaled_X, Y_train = pickle.load(file)

    return X_train, Y_train


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
                labels.append(int(label))
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def cnn_detect(X_train, X_val, Y_train, Y_val):
    '''Trains a simple convnet on the MNIST dataset.
    Gets to 99.25% test accuracy after 12 epochs
    (there is still a lot of margin for parameter tuning).
    16 seconds per epoch on a GRID K520 GPU.
    '''
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras.callbacks import ModelCheckpoint
    from keras import backend as K

    batch_size = 128
    num_classes = 9
    epochs = 12

    # input image dimensions
    img_rows, img_cols = 32, 32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_train /= 255
    X_val /= 255
    # print('x_train shape:', x_train.shape)
    # print(x_train.shape[0], 'train samples')
    # print(x_test.shape[0], 'test samples')

    Y_train = np.reshape(Y_train, (len(Y_train), 1))
    print(Y_train.shape)
    # convert class vectors to binary class matrices
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_val = keras.utils.to_categorical(Y_val, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                    activation='relu',
                    input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3),
                activation='relu',
                input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_val, Y_val),
            callbacks=callbacks_list)

    model.save("model.h5")
    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def svm(X_scaler, X_train, Y_train, X_val, Y_val):
    print('Start training')
    model = SVC(C=1, kernel='rbf')
    score = model.fit(X_train, Y_train)
    print(score)

    save_path = os.path.join(os.getcwd(), 'model.pkl')
    with open('model.pkl', 'wb') as file:
        pickle.dump([params, X_scaler, model], file)
        print('save data file at {}'.format(save_path))

    # with open('model.pkl', 'rb') as file:
    #     model = pickle.load(file)
    print(X_val.shape)
    # X_test_feats = extract_features(X_val, params)
    scaled_X = X_scaler.transform(X_val)
    Y_pred = model.predict(scaled_X)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(Y_pred, Y_val))
    return model

def main():
    X_data, Y_data = load_dataset()
   
    X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, random_state=1, test_size=0.3)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    print(Y_train.shape)
    print(X_train.shape)
    # svm(X_scaler, X_train, Y_train, X_val, Y_val)'
    cnn_detect(X_train, X_val, Y_train, Y_val)

if __name__ == '__main__':
    main()