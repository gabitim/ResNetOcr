from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow import keras

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

import matplotlib.pyplot as plt
from imutils import build_montages
import numpy as np
import cv2

EPOCHS = 3
INIT_LR = 1e-1
BS = 128


class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
                        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data

        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act1)

        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
                       padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                 momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
                       kernel_regularizer=l2(reg))(act3)

        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                              use_bias=False, kernel_regularizer=l2(reg))(act1)

        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])

        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
              reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(inputs)

        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply a single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias=False,
                       padding="same", kernel_regularizer=l2(reg))(x)

        # check to see if we are using the Tiny ImageNet dataset
        elif dataset == "tiny_imagenet":
            # apply CONV => BN => ACT => POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias=False,
                       padding="same", kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                                   momentum=bnMom)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                                           (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)

        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                               momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="resnet")

        # return the constructed network architecture
        return model


def load_mnist_dataset():
    # load data from tensorflow framework
    print('loading 0-9 dataset')
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

    # Stacking train data and test data to form single array named data
    data = np.vstack([trainData, testData])

    # Vertical stacking labels of train and test set
    labels = np.hstack([trainLabels, testLabels])

    # return a 2-tuple of the MNIST data and labels
    return data, labels


def load_az_dataset(datasetPath):
    print('loading A-Z dataset')

    # List for storing data
    data = []

    # List for storing labels
    labels = []

    for row in open(datasetPath):  # Openfile and start reading each row
        # Split the row at every comma
        row = row.split(",")

        # row[0] contains label
        label = int(row[0])

        # Other all collumns contains pixel values make a saperate array for that
        image = np.array([int(x) for x in row[1:]], dtype="uint8")

        # Reshaping image to 28 x 28 pixels
        image = image.reshape((28, 28))

        # append image to data
        data.append(image)

        # append label to labels
        labels.append(label)

    # Converting data to numpy array of type float32
    data = np.array(data, dtype='float32')

    # Converting labels to type int
    labels = np.array(labels, dtype="int")

    return data, labels


if __name__ == '__main__':

    '''
    1. Loading Datasets
    '''
    (digitsData, digitsLabels) = load_mnist_dataset()

    (azData, azLabels) = load_az_dataset('.\\Data\\A_Z Handwritten Data.csv')

    '''
    2. Combining datasets and dataset preparation
    '''
    # the MNIST dataset occupies the labels 0-9, so let's add 10 to every A-Z label to ensure the A-Z characters are
    # not incorrectly labeled

    azLabels += 10

    # stack the A-Z data and labels with the MNIST digits data and labels

    data = np.vstack([azData, digitsData])
    labels = np.hstack([azLabels, digitsLabels])

    # Each image in the A-Z and MNIST digts datasets are 28x28 pixels;
    # However, the architecture we're using is designed for 32x32 images,
    # So we need to resize them to 32x32

    data = [cv2.resize(image, (32, 32)) for image in data]
    data = np.array(data, dtype="float32")

    # add a channel dimension to every image in the dataset and scale the
    # pixel intensities of the images from [0, 255] down to [0, 1]

    data = np.expand_dims(data, axis=-1)
    data /= 255.0

    le = LabelBinarizer()
    labels = le.fit_transform(labels)

    counts = labels.sum(axis=0)

    # account for skew in the labeled data
    classTotals = labels.sum(axis=0)
    classWeight = {}

    # loop over all classes and calculate the class weight
    for i in range(0, len(classTotals)):
        classWeight[i] = classTotals.max() / classTotals[i]

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

    '''
    3. Performing Data Augmentation
    '''
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.05,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        fill_mode="nearest")

    '''
    4. Building ResNet Architecture
    '''

    '''
    5. Compiling model
    '''
    opt = SGD(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)

    model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                         (64, 64, 128, 256), reg=0.0005)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS, epochs=EPOCHS,
        class_weight=classWeight,
        verbose=1)

    '''
    6. Model Evaluation 
    '''
    labelNames = "0123456789"

    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    labelNames = [l for l in labelNames]

    predictions = model.predict(testX, batch_size=BS)

    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('OCR_Resnet.h5', save_format="h5")
    # construct a plot that plots and saves the training history
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('TrainingLossAndAccuracy.png')

    # randomly select a few testing characters
    images = []
    output = ""

    for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
        probs = model.predict(testX[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = labelNames[prediction[0]]
        output += label
        image = (testX[i] * 255).astype("uint8")
        color = (0, 255, 0)
        if prediction[0] != np.argmax(testY[i]):
            color = (0, 0, 255)
        image = cv2.merge([image] * 3)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    color, 2)
        images.append(image)

    montage = build_montages(images, (96, 96), (7, 7))[0]
    cv2.imwrite('ModelTestOutput.png', montage)
