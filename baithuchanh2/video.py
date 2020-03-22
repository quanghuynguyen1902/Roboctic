import cv2
import numpy as np
cap = cv2.VideoCapture(0)
i = 0
while True:
    _, frame = cap.read()
    if i % 10 ==0:
        cv2.imwrite('./image/data/frame%i.jpg'%i, frame)
    cv2.imshow("frame", frame)
    i += 1
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
