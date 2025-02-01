import cv2
import imutils

img = cv2.imread('img.jpeg')
imgresized = imutils.resize(img, width=100)


cv2.imshow("Original Image", img)
cv2.imshow("Resized Image", imgresized)

cv2.imwrite("Resizedimg.jpg",imgresized)

