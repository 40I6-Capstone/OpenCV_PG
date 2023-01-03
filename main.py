import cv2
import numpy as np

img = cv2.imread('square.jpg')
img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

cv2.namedWindow('HSV', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)

cv2.imshow('HSV',img_HSV)

result = img.copy()

# lower boundary BLUE color range values; Hue (100 - 140)
lower1 = np.array([90, 100, 0])
upper1 = np.array([150, 255, 255])

# upper boundary BLUE color range values; Hue (160 - 180)
lower2 = np.array([110, 120, 20])
upper2 = np.array([130, 255, 255])

lower_mask = cv2.inRange(img, lower1, upper1)
upper_mask = cv2.inRange(img, lower2, upper2)

full_mask = lower_mask + upper_mask

result = cv2.bitwise_and(result, result, mask=full_mask)

cv2.imshow('mask', full_mask)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Edge detection', cv2.WINDOW_NORMAL)
#
# canny_lower = 0
# canny_upper = 0
#
#
# def changeCannyLower(*args):
#     global canny_lower
#     canny_lower = args[0]
#
#
# def changeCannyUpper(*args):
#     global canny_upper
#     canny_upper = args[0]
#
#
# cv2.createTrackbar('Lower threshold', 'Edge detection', 0, 255, changeCannyLower)
# cv2.createTrackbar('Upper threshold', 'Edge detection', 0, 255, changeCannyUpper)
#
# def edge_detection():
#     edge = cv2.Canny(img, canny_lower, canny_upper)
#
#     cv2.imshow('Original Image', img)
#     cv2.imshow('Edge detection',edge)
#
# while True:
#     edge_detection()
#     cv2.waitKey(10)