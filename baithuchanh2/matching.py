import cv2
import numpy as np
import imutils

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

# Read the template 
template = cv2.imread('./image/template/sample1.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) 
template = cv2.Canny(template, 50, 200)
   
# Store width and height of template in w and h 
(tH, tW) = template.shape[:2]

  

while True:
    _, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (box_check[0], box_check[1]), (box_check[0] + box_check[2], box_check[1] + box_check[3]), (255, 0, 0), 3)

    found = None
    for scale in np.linspace(0.3, 1, 3)[::-1]: 
        # resize the image according to the scale, and keep track 
        # of the ratio of the resizing 
        resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale)) 
        r = img_gray.shape[1] / float(resized.shape[1]) 

        if resized.shape[0] < tH or resized.shape[1] < tW: 
            break
    
        # if the resized image is smaller than the template, then break 
        # from the loop 
        # detect edges in the resized, grayscale image and apply template  
        # matching to find the template in the image 
        edged  = cv2.Canny(resized, 50, 200) 
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF) 
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result) 
        # if we have found a new maximum correlation value, then update 
        # the found variable if found is None or maxVal > found[0]: 
        found = (maxVal, maxLoc, r) 

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r)) 
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r)) 
    
    box_object = (startX, startY, endX - startX, endY -startY)
    # draw a bounding box around the detected result and display the image 
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2) 

   
    if box_object != '':
        if(IOU(box_object, box_check) > 0.3):
            cv2.putText(frame, 'Calculate Detected', org, font, fontScale, color, thickness, cv2.LINE_AA) 

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()