import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    location = ''
    for i in range(6):
        template = cv2.imread("./image/template/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        print(i)
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.6)
        if len(loc[0]) != 0:
            location = zip(*loc[::-1])
            break

    # re = np.arange(10)
    # re = re[::-1]
    # print(re)
    # loc = np.where(re >= 5)
    # t = zip(*loc[::-1])
    # for i in t:
    #     print(i)
    print('done')
    for pt in location:
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()