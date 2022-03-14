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
        img = cv2.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images

def load_letters_from_folder(folder):
    letters = []
    for letter in os.listdir(folder):
        letters.append(letter)
    return letters

for letter in load_letters_from_folder("cnts"):
    for roi in load_images_from_folder(os.path.join("cnts",letter)):
        letter_keyboard_code = int(letter)
        roismall = cv2.resize(roi,(10,10))
        responses.append(letter_keyboard_code)
        sample = roismall.reshape((1,100))
        samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
