import cv2

img = cv2.imread('img.jpeg')

gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('original img.png',img)

cv2.imshow('grey img.png',gray_scale)
