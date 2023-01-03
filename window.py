import cv2
import numpy as np

kernel = np.ones((5, 5), np.uint8)
img = cv2.imread('DALL_E.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Grayscale', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge detection', cv2.WINDOW_NORMAL)
cv2.namedWindow('Opening', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)

canny_lower = 0
canny_upper = 120


def changeCannyLower(*args):
    global canny_lower
    canny_lower = args[0]


def changeCannyUpper(*args):
    global canny_upper
    canny_upper = args[0]


cv2.createTrackbar('Lower threshold', 'Edge detection', 0, 255, changeCannyLower)
cv2.createTrackbar('Upper threshold', 'Edge detection', 0, 255, changeCannyUpper)


def edge_detection():
    edge = cv2.Canny(img_blur, canny_lower, canny_upper, L2gradient=True)

    cv2.imshow('Original Image', img_blur)
    cv2.imshow('Edge detection', edge)
    mask = cv2.inRange(edge, canny_lower, canny_upper)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opening', opening)
    cv2.imshow('Grayscale', img_gray)

    # find the contours in the edged image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = img.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)
    print(len(contours), "objects were found in this image.")

    cv2.imshow("Contour", img_copy)


while True:
    edge_detection()
    cv2.waitKey(10)
