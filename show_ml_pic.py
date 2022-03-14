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
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
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
