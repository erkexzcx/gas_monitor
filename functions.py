#!/bin/python3

import cv2
import numpy as np

def process_image(img):
    # Rotate image
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), 4.7, 1.0) # +5 digrees
    img = cv2.warpAffine(img, M, (w, h))

    # Crop image
    img = img[730:783, 710:1065]

    # Extract only dark areas (text)
    l = np.array([75,75,75], dtype = "uint16")
    u = np.array([255,255,255], dtype = "uint16")
    img = cv2.inRange(img, l, u)

    # Remove small objects, which are definitely not numbers
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= 100:
            img[output == i + 1] = 255
    img = img.astype(np.uint8)

    return img

def invert_image(img):
    return (255-img)

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def load_ml_data():
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses.data', np.float32)
    responses = responses.reshape(responses.size, 1)

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE,responses)

    return model

def is_digit_contour(w, h):
    res = (21 > w > 10) and (37 > h > 30)
    #print ("w:" + str(w) + " h:" + str(h) + " res:" + str(res))
    return res
