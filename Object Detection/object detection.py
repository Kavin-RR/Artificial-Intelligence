import cv2
import imutils

cam = cv2.VideoCapture(0)

firstframe = None
area = 400

while True:
    _,vid = cam.read()
    text = 'Normal'

    vid = imutils.resize(vid , width = 1000)

    grayscale = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(grayscale,(21,21),2)


    if firstframe is None:
        firstframe = gaussian
        continue
    vid_dif = cv2.absdiff(firstframe,gaussian)

    threshold = cv2.threshold(vid_dif,25,255,cv2.THRESH_BINARY)[1]

    threshold = cv2.dilate(threshold,None,iterations = 2)

    cnts = cv2.findContours(threshold.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c)<area:
            continue

        (x,y,w,h)= cv2.boundingRect(c)
        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,255,0),2)
        text = "moving object deteted"
    print(text)

    cv2.putText(vid,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow('videocam',vid)

    key = cv2.waitKey(1)
    print(key)

    if key == ord('a'):
        break

cam.release()
cv2.destroyAllWindows
        
        

    
