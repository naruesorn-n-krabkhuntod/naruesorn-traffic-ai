from ultralytics import YOLO


# model = YOLO('yolov8n.yaml')
model = YOLO('../assets/model/yolov9t.pt')
path = 'P:\iplus.Traffic.Ai\datasets\helmet_v2.v1i.yolov9\data.yaml'
results = model.train(data=path, epochs=3)


results = model.val()

success = model.export(format='onnx')