import cv2
import numpy as np

def IOU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0 or h_intersection <= 0: # No overlap
        return 0
    I = w_intersection * h_intersection
    U = w1 * h1 + w2 * h2 - I # Union = Total Area - I
    return I / U

cap = cv2.VideoCapture(0)

box_object = ''
box_check = (480, 230, 150, 250)
font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 0.5
thickness = 1
color = (0, 0, 255) 
org = (480, 220) 

while True:
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (box_check[0], box_check[1]), (box_check[0] + box_check[2], box_check[1] + box_check[3]), (255, 0, 0), 3)
    
    location = ''
    for i in range(10):
        template = cv2.imread("./image/template/" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.6)
        if len(loc[0]) != 0:
            location = zip(*loc[::-1])
            break


    if location != '':
        pt = max(location)
        box_object = (pt[0], pt[1], w, h)
        print(box_object)
        cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

    if box_object != '':
        if(IOU(box_object, box_check) > 0.3):
            cv2.putText(frame, 'Calculate Detected', org, font, fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()