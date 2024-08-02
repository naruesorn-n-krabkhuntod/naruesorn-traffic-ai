import cv2
from ultralytics import YOLO

plate_model = YOLO("best.pt")

video = cv2.VideoCapture(0)
while True:
    _, im0 = video.read()
    tracks = plate_model.predict(im0, show=True)
    if cv2.waitKey(0) == 27 : break

cv2.destroyAllWindows()