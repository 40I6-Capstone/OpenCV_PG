import cv2
import numpy as np
import matplotlib.pylab as plt

bck = cv2.imread("square_small.jpg")
img = cv2.imread("square_small_bg.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('Final', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)

mask = cv2.inRange(hsv, (0, 0, 0), (0, 0, 75))
cv2.imshow('Mask', mask)

imask = mask>0
orange = np.zeros_like(img, np.uint8)
orange[imask] = img[imask]

yellow = img.copy()
hsv[...,0] = hsv[...,0] + 20
yellow[imask] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[imask]
yellow = np.clip(yellow, 0, 255)

bckfish = cv2.bitwise_and(bck, bck, mask=imask.astype(np.uint8))
nofish = img.copy()
nofish = cv2.bitwise_and(nofish, nofish, mask=(np.bitwise_not(imask)).astype(np.uint8))
nofish = nofish + bckfish

nofish_rgb = cv2.cvtColor(nofish, cv2.COLOR_BGR2RGB)
cv2.imshow('Final',nofish_rgb)
cv2.waitKey(0)
