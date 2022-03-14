#!/bin/python3

from functions import *

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()

img = process_image(img)
img = invert_image(img)

cv2.imshow("res", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
