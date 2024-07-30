import cv2
import numpy as np


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),5,(0,0,255),-1)
        cv2.imshow('image',img)
        mouseX,mouseY = x,y
        print (mouseX,mouseY)


vid = cv2.VideoCapture('thai_traffic.mp4')
ret, img = vid.read()
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(0) == 27 : break