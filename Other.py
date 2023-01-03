import cv2

image = cv2.imread('square_small.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

def changeCannyLower(*args):
    global canny_lower
    canny_lower = args[0]

def changeCannyUpper(*args):
    global canny_upper
    canny_upper = args[0]

cv2.createTrackbar('Lower threshold', 'Edge detection', 0, 255, changeCannyLower)
cv2.createTrackbar('Upper threshold', 'Edge detection', 0, 255, changeCannyUpper)

edged = cv2.Canny(blurred, 0,200, apertureSize=3, L2gradient = True)

# define a (3, 3) structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# apply the dilation operation to the edged image
dilate = cv2.dilate(edged, kernel, iterations=1)

# find the contours in the dilated image
contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
# draw the contours on a copy of the original image
cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
print(len(contours), "objects were found in this image.")

cv2.imshow("Dilated image", dilate)
cv2.imshow("contours", image_copy)
cv2.waitKey(0)

# cv2.imshow("Original image", image)
# cv2.imshow("Edged image", edged)
#
# # find the contours in the edged image
# contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# image_copy = image.copy()
# # draw the contours on a copy of the original image
# cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
# print(len(contours), "objects were found in this image.")
#
# cv2.imshow("contours", image_copy)
# cv2.waitKey(0)