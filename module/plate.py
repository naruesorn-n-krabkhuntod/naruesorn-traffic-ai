import cv2
import easyocr
from ultralytics import YOLO, solutions

ocr = easyocr.Reader(['th'], gpu=False)
plate_model = YOLO("assets/model/plate_yolov8n.pt")


def readPlateNumber(source):
    tracks = plate_model.track(source, persist=True, show=False)
    for track in tracks:
        _x, _y, _w, _h = int(track.boxes.xywh[0][0]), int(track.boxes.xywh[0][1]), int(track.boxes.xywh[0][2]), int(track.boxes.xywh[0][3])
        _x, _y = int(_x-_w/2), int(_y-_h/2)
    image = source[_y:_y+_h, _x:_x+_w]
    platenum = ocr.readtext(image, detail=False)
    print('\n\n\n\n', platenum, '\n\n\n\n')
    return (_x,_y,_w,_h), image


im0 = cv2.imread('export/counting/5.jpg')
(x,y,w,h), im1 = readPlateNumber(im0)
print(x,y,w,h)
cv2.imshow('test', im1)
cv2.waitKey(0) == 27
cv2.destroyAllWindows()