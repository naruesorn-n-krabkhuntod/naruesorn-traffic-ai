from ultralytics import YOLO


# model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
path = '/home/naruesorn/Desktop/yolov8-thaitrafiic-analysis/datasets/face-detection.v1i.yolov8/data.yaml'
results = model.train(data=path, epochs=3)


results = model.val()

success = model.export(format='onnx')