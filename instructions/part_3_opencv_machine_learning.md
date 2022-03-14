# Table of contents

- [1. Understanding the steps](#understanding-the-steps)
- [2. Cleanup your code](#cleanup-your-code)
- [3. Add additional functions to functions.py](#add-additional-functions-to-functionspy)
- [4. Adjust is_digit_contour function](#adjust-is-digit-contour-function)
- [5. Taking screenshots for first AI model](#taking-screenshots-for-first-ai-model)
- [6. Train AI manually](#train-ai-manually)
- [7. Save a lot of screenshots.](#save-a-lot-of-screenshots)
- [8. Use AI to sort digits into a folders structure](#use-ai-to-sort-digits-into-a-folders-structure)
- [9. Train actual AI model](#train-actual-ai-model)
- [10. Validate AI model](#validate-ai-model)

# Understanding the steps

Here is the list of steps that we will be performing. Understanding the steps will help you to understand what we are trying to achieve:
1. Create few screenshots that contains all possible digits (`0123456789`) at least once.
2. Manually train the AI model with that data.
3. Create A LOT of screenshots that contains all possible digits (`0123456789`), each digit will be found multiple times and be visible in various lighting conditions.
4. Use already trained AI model to extract those digits and put them into folders structures (e.g. `cnts/0/<images>`, `cnts/1/<images>` and so on)
5. Fix the mistakes made by already trained AI model (move incorrecly classified images to their corresponding folders)
6. Automatically train AI model with data in folders structure
7. Validate AI model

Note regarding steps 1 and 2 - you will train a pretty stupid AI model that will read some digits incorrecly (e.g. mostly confuses 6, 9 and 0), but the point of this model is to help you put digits into their folder structure for proper training. When you have 300 images and 2100 digits, it's a bit time consuming to move everything manually. Instead, we use this approach to speed up folders structure creation.

# Cleanup your code

Create a new file called `functions.py` and move functions from `test.py`. I've ended up with this `functions.py` file:
```python
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
```

And now your `test.py` file will look like this (much more readable):
```python
#!/bin/python3

from functions import *

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()
img = process_image(img)
img = reverse_image(img)

cv2.imshow("res", img)
cv2.waitKey(0)
```

# Add additional functions to functions.py

Add below additional functions to `functions.py` file. You will definitely need to update `is_digit_contour` function, probably leave `load_ml_data` function unmodified and definitely leave `sort_contours` function unmodified.
```python
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
    # print ("w:" + str(w) + " h:" + str(h)) # comment out this line if not needed
    return (21 > w > 10) and (37 > h > 30)
```

# Adjust is_digit_contour function

In `functions.py` file, there is function `is_digit_contour` which uses width and height to filter out digits from the image. In my case, each digit was between 10 and 21 pixels width and from 30 to 37 pixels height. Uncomment `print` line in that function to enable width and height logs, so you can have an idea of the dimensions of the digits.

Then create new file `test_digit_dimensions.py` with the following content:
```python
#!/bin/python3

from functions import *

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()

img = process_image(img)

# Find contours
cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
sorted_cnts, _ = sort_contours(cnts)

img = invert_image(img)

for c in sorted_cnts:
    x,y,w,h = cv2.boundingRect(c)
    if is_digit_contour(w, h):
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)

cv2.imshow("res", img)
cv2.waitKey(0)
```

And execute it. It will show you detected digits and their dimensions in the output:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic7.png)

Adjust values in function `is_digit_contour` accordingly, so it picks only digits and not anything else.

# Taking screenshots for first AI model

Create new folder `snapshots`. 

Then create a new file `save_snapshot.py` with the following contents:
```python
#!/bin/python3

import cv2
from datetime import datetime

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()
cv2.imwrite("snapshots/" + datetime.now().strftime("%m%d%Y%H%M%S") + ".jpg", img)
```

Each time you execute this file, it will take frame from the camera stream and saves it to `snapshots/<image>`.

Using this script save only a few images where each digit will be found in images at least once.

**Note**: Instead of using Python script (you will need this exact script later), for now feel free to open video stream using VLC and press `Alt+v` and then `Alt+s` to save frame to a `~/Pictures` folder. Later on just move those pictures to `snapshots` directory (file name does not matter).

# Train AI manually

Create a new file `train_model_local.py` with the following contents:
```python
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

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)
```

And execute it. It will show each digit and waits for your keypress matching the digit (for number 9 press `9` on your keyboard) and continue it until program exits (after reading all saved images in `snapshots` folder). Once done, you will have 2 new files `generalsamples.data` and `generalresponses.data` containing the AI model data.

# Save a lot of screenshots.

From the previous step you have created a file `save_snapshot.py`. Use it to create a lot of images (e.g. at least 1000 of them).

**Example 1**: using `watch` command (executes as fast as it can, in my case every 2-3 seconds):
```
watch -n 1 python3 save_snapshot.py
```

And leave above command for e.g. 1-2 hours (ensure all digits are saved, each multiple times)

**Example 2**: using crontab - leave overnight to capture a lot of images:
```
* * * * * cd /path/to/working/directory && python3 save_snapshot.py
```

**Note**: For better accuracy, take some pictures using different lightning conditions (day/night, light on/off in the room).

# Use AI to sort digits into a folders structure

Create new file `parse_tofolders.py` with the following contents:
```python
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
```

And execute it. Once script is finished, you will find a new directory `cnts` containing each digit folder and inside the pictures of matching digit.

Now a boring part - you will need to manually move files between digit folders. It is recommended to use file explorer (file manager) to see previews as it makes much easier to see pictures in wrong directories. For example, there is a picture of a digit `9` in a folder `cnts/0`, so manually move it to `cnts/9`. Ensure all images are moved to their corresponding folders and all images are matching their parent directory name.

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic8.png)

# Train actual AI model

Delete files `generalsamples.data` and `generalresponses.data` and create new file `train_model_fromfolders.py` with the following content:
```python
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
```

Then execute it and you will find a new files `generalsamples.data` and `generalresponses.data` that contain actual AI model data generated from `cnts` directories.

# Validate AI model

Create a new file `show_ml_pic.py` using below content:
```python
#!/bin/python3

from functions import *

model = load_ml_data()

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()

img = process_image(img)

# Find contours
img = img.astype(np.uint8)
cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
sorted_cnts, _ = sort_contours(cnts)

img = invert_image(img)

digits_string = ""
for c in sorted_cnts:
    x,y,w,h = cv2.boundingRect(c)
    if is_digit_contour(w, h):
        roi = img[y:y+h,x:x+w]
        roismall = cv2.resize(roi,(10,10))
        roismall = roismall.reshape((1,100))
        roismall = np.float32(roismall)
        retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
        digit_str = str(int((results[0][0])))

        digits_string += digit_str

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 1)
        cv2.putText(img, digit_str, (x+15, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

print(digits_string)

cv2.imshow("res", img)
cv2.waitKey(0)
```

and execute it. You will get a similar picture:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic9.png)
