from matplotlib import pyplot as plt
from skimage import measure
# Find contours at a constant value of 10
contours = measure.find_contours(image='square_small.jpg', level=10)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow('square_small.jpg', cmap=plt.cm.gray)

for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()