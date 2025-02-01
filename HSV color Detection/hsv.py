import cv2
import imutils

redlow =(0,184,204)
redhigh=(87,179,255)

cam = cv2.VideoCapture(0)

while True:

    _,vid = cam.read()

    frame = imutils.resize(vid, width = 750)

    gaussian = cv2.GaussianBlur(frame,(11,11),0)

    hsv = cv2.cvtColor(gaussian,cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,redlow,redhigh)
    mask=cv2.erode(mask,None,iterations = 2)
    mask=cv2.dilate(mask,None,iterations = 2)

    cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    center = None

    if len(cnts) > 0:
        c = max(cnts , key = cv2.contourArea)
        ((x,y),radius) =cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] != 0:  # Check to avoid division by zero
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (0, 0)  # Default or handle zero moments case
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame,center,5,(0,0,255),-1)
            print(center,radius)

            if radius > 250:
                print("stop")
            else:
                if (center[0]<150):
                    print("right")
                elif (center[0]>450):
                    print('left')
                elif(radius < 250):
                    print("front")
                else:
                    print("stop")
    cv2.imshow('frame',frame)
    Key = cv2.waitKey(1)
    if Key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows
