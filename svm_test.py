import cv2
import numpy as np

hog = cv2.HOGDescriptor()
hog.load('myHogDector.xml')
img = cv2.imread('test.jpg')
#img = cv2.resize(img, (256, 150))
rects, wei = hog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8), scale=1.05)
for x, y, w, h in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
print(wei)
img = cv2.resize(img, (700, 500))
cv2.imshow("hog", img)
cv2.waitKey(0)