import cv2
import numpy as np

img = cv2.imread("./image/data/frame30.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
location = ''
for i in range(6):
    template = cv2.imread("./image/template/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]

    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= 0.7)
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
for pt in location:
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

cv2.imshow("img", img)


cv2.waitKey(0)
cv2.destroyAllWindows()