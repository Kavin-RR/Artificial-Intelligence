import cv2
import numpy as np
import imutils

def nothing(x):
    # Callback function for trackbars (does nothing but is required)
    pass

# Create a window
cv2.namedWindow("Color Calibration")

# Create trackbars for color calibration in HSV space
cv2.createTrackbar("Lower H", "Color Calibration", 0, 179, nothing)
cv2.createTrackbar("Lower S", "Color Calibration", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Color Calibration", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Color Calibration", 179, 179, nothing)
cv2.createTrackbar("Upper S", "Color Calibration", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Color Calibration", 255, 255, nothing)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    frame = imutils.resize(frame , width = 400)

    if not ret:
        break

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current trackbar positions
    lh = cv2.getTrackbarPos("Lower H", "Color Calibration")
    ls = cv2.getTrackbarPos("Lower S", "Color Calibration")
    lv = cv2.getTrackbarPos("Lower V", "Color Calibration")
    uh = cv2.getTrackbarPos("Upper H", "Color Calibration")
    us = cv2.getTrackbarPos("Upper S", "Color Calibration")
    uv = cv2.getTrackbarPos("Upper V", "Color Calibration")

    # Define the HSV range based on trackbar positions
    lower_bound = np.array([lh, ls, lv])
    upper_bound = np.array([uh, us, uv])

    # Create a mask and apply it to the frame
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame, mask, and result
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Result", result)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
