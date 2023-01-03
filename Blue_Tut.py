import cv2
import numpy

# cam = cv2.VideoCapture(0)
cam = cv2.imread('square_small.jpg', cv2.IMREAD_UNCHANGED)
kernel = numpy.ones((5, 5), numpy.uint8)

frame = cv2.imread('square_small.jpg', cv2.IMREAD_UNCHANGED)
rangomax = numpy.array([70,70,70])  # B, G, R
rangomin = numpy.array([0,0,0])
mask = cv2.inRange(frame, rangomin, rangomax)
# reduce the noise
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

x, y, w, h = cv2.boundingRect(opening)

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
# cv2.circle(frame, (x + w / 2, y + h / 2), 5, (0, 0, 255), -1)

cv2.imshow('camera', frame)

k = cv2.waitKey(0)

