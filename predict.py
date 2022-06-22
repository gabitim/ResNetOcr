import numpy as np
from imutils.contours import sort_contours
import imutils
import cv2
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("OCR_Resnet.h5")
print("model is loaded")


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def get_upper_lower_bound_from_array(array):
    # https://builtin.com/data-science/how-to-find-outliers-with-iqr

    array.sort()
    print(array)
    np_arr = np.array(array)

    q1, q3 = np.percentile(np_arr, [25, 75])
    print(q1, q3)

    IQR = q3 - q1

    lower_bound = (q1 - 1.5 * IQR) - 2
    print(lower_bound)

    return lower_bound


def get_height_upper_lower_bound(cnts):
    height_arr = []
    area_arr = []

    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        area_arr.append(w * h)
        if w * h > 100:
            height_arr.append(h)

    print(area_arr)

    return get_upper_lower_bound_from_array(height_arr)


if __name__ == "__main__":
    red_color = (0, 0, 255)
    file_path = './images/21_5.jpg'

    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    cv2.imshow('dilate_edged', resize_with_aspect_ratio(dilate, height=200))

    cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    chars = []

    h_lower_bound = get_height_upper_lower_bound(cnts)

    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out bounding boxes
        if h_lower_bound <= h:
            # extract the character and threshold it to make the character
            # appear as *white* (foreground) on a *black* background, then
            # grab the width and height of the thresholded image
            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), red_color, 1)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the
            # width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            # otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            # pad the image and force 32x32 dimensions
            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            cv2.imwrite(f"extracted_chars/{x}-{y}-{w}-{h}.jpg", padded)

            # prepare the padded image for classification via our
            # handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)
    # define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    output = ""
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        output += label

    print("output", output)

    cv2.imshow('chars bounding boxes', resize_with_aspect_ratio(image, height=200))
    cv2.waitKey(0)
