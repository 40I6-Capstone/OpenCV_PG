import cv2
import numpy as np
import math

global debug
debug = False

# Set Pipeline to True to see image pipeline
global Pipeline
Pipeline = True


def nothing(x):
    pass

def apply_brightness_contrast(input_img, brightness=0, contrast=0):  # function to change brightness and contrast in the image
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Contour', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('mask_result', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edge', cv2.WINDOW_NORMAL)
cv2.namedWindow('High Contrast', cv2.WINDOW_NORMAL)
cv2.namedWindow('Sharpened', cv2.WINDOW_NORMAL)

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
    frame = cv2.imread("Tello_test_images/20_newbg.png")


    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)

    frame_high_contrast = apply_brightness_contrast(image_sharp, 0, 50)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # min = [0,0,0]
    # max = [180,255,255]

    # Get Lower and Upper HSV+Canny values from the trackbars
    if debug:
        l_h = 30
    else:
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

    # Mask is applied to hsv of orignal image, using values selected on the trackbar
    mask = cv2.inRange(hsv, lower, upper)
    invert_mask = cv2.bitwise_not(mask)  # inverted version of the mask

    # mask result (selecting only the oil spill from the original image)
    result = cv2.bitwise_and(frame, frame, mask=invert_mask)

    # apply a Gaussian Blur to make the contours easier to detect
    img_blur = cv2.GaussianBlur(result, (17, 17), 0)
    edge = cv2.Canny(img_blur, canny_lower, canny_upper, L2gradient=True)

    # find the contours in the inverted mask
    contours, _ = cv2.findContours(invert_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_copy = frame.copy()

    # Find the largest Contour and draw a bounding box
    try:
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)  # (x,y) is the top-left coordinate of the rectangle and (w,h) is its width and height.

        # red color in BGR
        red = (0, 0, 255)
        blue = (255, 0, 0)
        green = (0, 0, 255)
        orange = (0, 123, 255)

        # draw the bounding rectangle for the biggest contour (c) in red
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), red, 15)

        # Find the index of the largest contour
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        # # draw the largest contour in blue
        cv2.drawContours(img_copy, contours, max_index, blue, 15)

    except:  # Prevents code from crashing when upper and lower limits are all set to 0 (i.e. trackbars not modified)
        print("no contour found")

    # draw the contours on a copy of the original image
    cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

    # Display
    cv2.imshow("Original Image", frame)
    cv2.imshow("Contour", img_copy)
    if Pipeline:
        cv2.imshow("High Contrast", frame_high_contrast)
        cv2.imshow("mask", mask)
        cv2.imshow("mask_result", result)
        cv2.imshow('Edge', edge)
        cv2.imshow('Sharpened',image_sharp)

    # wait for a key to pressed, if not then close
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
