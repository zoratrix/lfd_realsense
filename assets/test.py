from ultralytics import YOLO
model = YOLO('perception/runs/perception_v12/weights/best.pt')
print(model.names)