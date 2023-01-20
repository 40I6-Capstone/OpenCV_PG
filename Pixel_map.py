import cv2

# Load the image
img = cv2.imread("Tello_test_images/40.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find the contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through the contours
for contour in contours:
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # Get the number of vertices of the polygon
    vertices = len(approx)

    # Determine the shape of the contour
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices == 6:
        shape = "Hexagon"
    else:
        shape = "Circle"

    # Draw the contour on the original image
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)

    # Get the center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Put the shape name on the image
    cv2.putText(img, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Show the image with the shapes
cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()