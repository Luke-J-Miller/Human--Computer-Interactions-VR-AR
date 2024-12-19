!pip install ultralytics

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
success = model.export(format="onnx")
