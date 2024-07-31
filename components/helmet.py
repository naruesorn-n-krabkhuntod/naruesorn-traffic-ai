import cv2
from ultralytics import YOLO

plate_model = YOLO("../assets/model/best.pt")


while True:
    im0 = cv2.imread('../' + input())
    tracks = plate_model.track(im0, persist=True, show=True)
    if cv2.waitKey(0) == 27 : break

cv2.destroyAllWindows()