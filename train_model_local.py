#!/bin/python3

import os
import sys

from functions import *

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

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
            roi = img[y:y+h, x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',roismall)
            key = cv2.waitKey(0)
            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print("training complete")

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
