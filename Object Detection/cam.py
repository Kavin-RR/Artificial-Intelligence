import cv2

vs = cv2.VideoCapture(0)

while True :
    _,img = vs.read()

    cv2.imshow('lapcam',img)

    key =cv2.waitKey(1)

    print(key)

    if key == ord ("a"):
        break

vs.release()
cv2.destroyAllWindows
