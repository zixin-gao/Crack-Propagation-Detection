import cv2
import numpy as np                  # mathematical calculations + matrices
import matplotlib.pyplot as plt     # helps plot graphs
import time                         # display time
import imutils                      # helps process images

# load image
img = cv2.imread("z__references/image1.jpg")

# convert to grey scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(gray1, (40, 20))
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# detect contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours on the original image
# cv2.drawContours(img, contours, -1, (0, 255, 0), 5)

# display windows
cv2.namedWindow("Thresh Image", cv2.WINDOW_NORMAL)
cv2.imshow("Thresh Image", thresh)

# cv2.namedWindow("Contour Image", cv2.WINDOW_NORMAL)
# cv2.imshow("Contour Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()