import cv2

img = cv2.imread('img.jpeg')

grayscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshold = cv2.threshold(grayscale,100,255,cv2.THRESH_BINARY)[1]

cv2.imshow('original img.png',img)
cv2.imshow('grayimg.png',grayscale)
cv2.imshow('thresholdimg.png',threshold)
