import cv2
import numpy as np
img = cv2.imread('img.jpeg')

#recimg = cv2.rectangle(img,(10,10),(10+50,10+50),(0,255,0),2)

#cv2.imshow('orgimg',img)
#cv2.imshow('rect img',recimg)

#text = "hello i am kavin"
#textimg = cv2.putText(img,text,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

#cv2.imshow('textimg.ong',textimg)

cntimg = cv2.findContours(thresholdimg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow('cntsimg',cntimg)
