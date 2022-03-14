#!/bin/python3

import cv2
from datetime import datetime

cap = cv2.VideoCapture('http://172.18.17.68:8081/0/stream')
_, img = cap.read()
cv2.imwrite("snapshots/" + datetime.now().strftime("%m%d%Y%H%M%S") + ".jpg", img)
