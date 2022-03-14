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
