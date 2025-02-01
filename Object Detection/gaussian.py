import cv2

img = cv2.imread('img.jpeg')

gaussianimg1= cv2.GaussianBlur(img ,(41,41) , 50)
gaussianimg2= cv2.GaussianBlur(img ,(21,21) , 0)


cv2.imshow('original.png',img)
cv2.imshow('blurimg1.png',gaussianimg1)
cv2.imshow('blurimg2.png',gaussianimg2)

