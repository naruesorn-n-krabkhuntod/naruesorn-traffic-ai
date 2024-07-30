# import module
import os, cv2, time
import numpy as np
from ultralytics import YOLO, solutions


# load model
model = YOLO("assets/model/yolov9s.pt")

# load source
cap = cv2.VideoCapture("assets/video/sample.mp4")
assert cap.isOpened(), "Error reading video file"


# detect speed
speed_obj = solutions.SpeedEstimator(
    reg_pts=[(0, 350), (1280, 350)],
    names=model.model.names,
    view_img=False,
    spdl_dist_thresh=5
)


# write detection line
success, im_detect = cap.read()
# cv2.rectangle(im_detect, (0,0), (im_detect.shape[1], im_detect.shape[0]), (0,0,0), -1)
cv2.line(im_detect, (673,0), (312,720), (0,0,255), 10)
cv2.line(im_detect, (0,610), (1280,610), (255,0,255), 30)
cv2.line(im_detect, (210,400), (990,400), (255,255,0), 30)
cv2.fillPoly(im_detect, [np.array([[945, 280],[1018,440], [1256,440], [1116, 280]])], (0,255,255))
# cv2.imshow("iplus Traffic Analysis", im_detect)
# cv2.waitKey(0)


# other setup variable
change_len_list = []
counting_list = []
speed_time = {}
speed_list = {}
counting = 0

while cap.isOpened():
    success, im0 = cap.read()
    im1 = im0.copy()
    im_def = im0.copy()
    if not success or cv2.waitKey(1) == 27 : break
    tracks = model.track(im0, persist=True, show=False)
    im0 = speed_obj.estimate_speed(im0, tracks)


    # overlay
    cv2.polylines(im1, [np.array([[180, 420],[10,605], [361,605], [451, 420]])], True, (0,0,255), 2)
    cv2.polylines(im1, [np.array([[476, 420],[382,605], [734,605], [735, 420]])], True, (0,0,255), 2)
    cv2.polylines(im1, [np.array([[776, 420],[783,605], [1065,605], [992, 420]])], True, (0,0,255), 2)
    cv2.polylines(im1, [np.array([[945, 280],[1018,440], [1256,440], [1116, 280]])], True, (255,0,0), 2)
    cv2.putText(im1, str(counting), (40,40), 1, 3, (0,0,255), 4)



    if speed_obj.boxes is not None:
        for box, clss, id in zip(speed_obj.boxes, speed_obj.clss, speed_obj.trk_ids):
            if int(clss) == 0 : continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            center_x = np.clip(((x1 + x2) // 2), 1, 1279)
            center_y = np.clip((y2), 1, 719)
            center_color = im_detect[center_y, center_x]

            # write Detection
            if int(clss) == 3 : y1 = np.clip(y1 - ((y2 - y1 - 10) // 2), 0, 720)
            if id in change_len_list : cv2.rectangle(im1, (x1,y1), (x2, y2), (0,0,255), 2)
            elif id in counting_list : cv2.rectangle(im1, (x1,y1), (x2, y2), (0,255,255), 2)
            elif id in speed_time : cv2.rectangle(im1, (x1,y1), (x2, y2), (255,0,0), 2)
            else : cv2.rectangle(im1, (x1,y1), (x2, y2), (0,255,0), 2)
            cv2.putText(im1, str(speed_obj.names[clss]), (x1,y1+15), 1, 1.2, (0,0,255), 2)
            cv2.putText(im1, str(id), (x1,y1+30), 1, 1.2, (0,0,255), 2)
            cv2.circle(im1, (center_x, center_y), 2, (0,0,255), -1)
            if id in speed_list : cv2.putText(im1, str(speed_list[id]), (x1,y1+45), 1, 1.2, (0,0,255), 2)


            # detect len change
            if([center_color[0], center_color[1], center_color[2]] == [0, 0, 255]):
                if y2 <= 700 : 
                    imcoppy = im_def.copy()
                    cv2.circle(imcoppy, (center_x, np.clip(((y1 + y2) // 2), 1, 1279)), y2-y1+5, (0,0,255), 2)
                    if not id in change_len_list: change_len_list.append(id)
                    # cv2.imshow("detect", imcoppy)
                    cv2.imwrite(os.path.join("./export/changeline/" , str(id) + ".jpg"), imcoppy)

            # detect footbath
            elif([center_color[0], center_color[1], center_color[2]] == [0, 255, 255]):
                imcoppy = im_def.copy()
                cv2.circle(imcoppy, (center_x, np.clip(((y1 + y2) // 2), 1, 1279)), y2-y1+5, (0,0,255), 2)
                if not id in change_len_list: change_len_list.append(id)
                # cv2.imshow("detect", imcoppy)
                cv2.imwrite(os.path.join("./export/footpath/" , str(id) + ".jpg"), imcoppy)

            # speed start
            elif([center_color[0], center_color[1], center_color[2]] == [255, 255, 0]):
                if not id in speed_time : speed_time[id] = time.time()
                # print(speed_time)
            
            # counting
            elif([center_color[0], center_color[1], center_color[2]] == [255, 0, 255]) and (not id in counting_list):
                if id in speed_time : speed_list[id] = 0.05 / ((time.time() - speed_time[id]) / 3600)
                counting_list.append(id)
                counting += 1
                imcoppy = im_def[y1 : y2, x1 : x2]
                cv2.imwrite(os.path.join("./export/counting/" , str(counting) + ".jpg"), imcoppy)


        cv2.imshow("iplus Traffic Analysis", im1)   
        if cv2.waitKey(1) == 27 : exit()
    

cap.release()
cv2.destroyAllWindows()