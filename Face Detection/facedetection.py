import cv2

alg = 'haarcascade_frontalface_default.xml'

haar_cascade = cv2.CascadeClassifier(alg)

cam = cv2.VideoCapture(0)

while True:

    _,vid = cam.read()

    grayscale = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)

    face = haar_cascade.detectMultiScale(grayscale,1.3,4)

    for (x,y,w,h) in face :

        cv2.rectangle(vid,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow('face detectin',vid)

    key = cv2.waitKey(10)

    if key == 27:
        break

cv2.release()
cv2.destroyAllWindows


