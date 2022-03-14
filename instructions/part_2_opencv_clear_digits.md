# Test if OpenCV works:

Create a new file called `test.py` using below contents (change stream URL):
```python
#!/bin/python3

import cv2
import numpy as np

def process_image(img):
    # More code goes here
    return img

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()
img = process_image(img)

cv2.imshow("res", img)
cv2.waitKey(0) # Press any **keyboard** key to exit
```

And execute it (`python3 test.py`). You should get a single frame of a video stream. Example:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic4.png)

# Extract digits display only

Now add below code to `# More code goes here` section in order to rotate and crop image. Adjust below configuration or even remove unnecesarry code:
```python
    # Rotate image
    rotation = 4.7 # Can be negative
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Crop image
    img = img[730:783, 710:1065]
```

And you will end up with something like this:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic5.png)

Note the following:
* White digits are clearly distinct from dark background (black, brown or red - doesn't matter as long as it's dark)
* Comma is very small, so we don't really mind it (later it will be removed).

# Show only digits

Append below code after the previously added code (in `# More code goes here`) in order to turn image to pure black & white mode, then remove small contours and reverse image (so digits are black, background is white):

```python
    # Turn image into pure black & white with threshold
    l = np.array([75,75,75], dtype = "uint16")    # White colour stars from about [75,75,75] (small particles/noise will be removed later)
    u = np.array([255,255,255], dtype = "uint16") # For white digits, leave [255,255,255] (maximum white)
    img = cv2.inRange(img, l, u)

    # Remove small objects, which are definitely not digits
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= 100: # Keep only contours which are larger than 100 black pixels
            img[output == i + 1] = 255
    img = img.astype(np.uint8)

    # Reverse image (digits are black, background is white)
    img = (255-img)
```

And you will end up with something like this:

![](https://github.com/erkexzcx/gas_monitor/raw/main/instructions/images/pic6.png)

Note the following:
* Small details/lines are not good, but they are fine if they span across whole image since we will filter out digits later on.
* (Some) digits are not missing, do not have missing parts and are overall human-readable.

# Conclusion

Here is the complete code I've used for the last image:
```python
#!/bin/python3

import cv2
import numpy as np

def process_image(img):
    # Rotate image
    rotation = 4.7 # Can be negative
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    # Crop image
    img = img[730:783, 710:1065]

    # Turn image into pure black & white with threshold
    l = np.array([75,75,75], dtype = "uint16")    # White colour stars from about [75,75,75] (small particles/noise will be removed later)
    u = np.array([255,255,255], dtype = "uint16") # For white digits, leave [255,255,255] (maximum white)
    img = cv2.inRange(img, l, u)

    # Remove small objects, which are definitely not digits
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= 100: # Keep only contours which are larger than 100 black pixels
            img[output == i + 1] = 255
    img = img.astype(np.uint8)

    return img

def reverse_image(img):
    return (255-img)

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()
img = process_image(img)
img = reverse_image(img)

cv2.imshow("res", img)
cv2.waitKey(0)
```
