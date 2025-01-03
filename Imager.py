import cv2
import numpy as np


def polygonator(image):
    
    img = cv2.imread(image)
    img = cv2.resize(img, (0,0), fx=3, fy=3)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray_img, 1000, 0.1, 20)
    corners = np.int64(corners)
    polygon =[]

    for c in corners:
        x, y = c.ravel()
        tlist = [int(x), int(y)]
        cv2.circle(img, (x, y), 5, (255, 0, 0), thickness = -1)
        polygon.append(tlist)

    return polygon

'''polygonator("CentroidSkier/Outline.png")

cv2.imshow('Skier', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''