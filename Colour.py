import cv2
import numpy as np

cap = cv2.VideoCapture(0) # webcam capture
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask_result', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)

# try increasing contrast and/or inverting colors to help bounding
# use contours method to select largest shape from mask result - increase lieniency
# don't blur the image to take care of the high spots
# also worth trying: as part of setup, take easier images with better lighting
# do the contour and edge on the mask, not the mask result - can use blur on the mask as well

def nothing(x):
    pass

# create trackbars to edit HSV lower and upper values for the mask
# also create trackbars to play with canny upper and lower thresholds
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
cv2.createTrackbar('Canny Lower', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Canny Upper', 'Trackbars', 0, 255, nothing)

while True:
    # _, frame = cap.read()
    frame = cv2.imread("square.jpg")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # min = [0,0,0]
    # max = [180,255,255]

    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    canny_lower = cv2.getTrackbarPos("Canny Lower", "Trackbars")
    canny_upper = cv2.getTrackbarPos("Canny Upper", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    invert_mask = cv2.bitwise_not(mask)

    result = cv2.bitwise_and(frame, frame, mask=invert_mask)

    img_blur = cv2.GaussianBlur(result, (17, 17), 0)
    edge = cv2.Canny(img_blur, canny_lower, canny_upper, L2gradient=True)

    # find the contours in the edged image
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = frame.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Original Image", frame)
    cv2.imshow("mask", mask)
    cv2.imshow("mask_result", result)
    cv2.imshow('Edge', edge)
    cv2.imshow("Contour", img_copy)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
