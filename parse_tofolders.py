#!/bin/python3

from pathlib import Path
import time
import os

from functions import *

model = load_ml_data()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

for img in load_images_from_folder("snapshots"):
    img = process_image(img)
    img = invert_image(img)

    # Find each symbol
    img = img.astype(np.uint8)
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if is_digit_contour(w, h):
            roi = img[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            digit_str = str(int((results[0][0])))

            Path("cnts/" + digit_str).mkdir(parents=True, exist_ok=True)
            cv2.imwrite("cnts/" + digit_str + "/" + str(time.time()) + ".jpg", roi)
