from ultralytics import YOLO
import sys

sys.path.append('/data/seekyou/Algos/ultralytics')
print(1)
model = YOLO(r"/data/seekyou/Algos/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml")

model._new("/data/seekyou/Algos/ultralytics/ultralytics/cfg/models/v8/yolov8n-obb.yaml",task="detect",verbose=True)
print(1)
print(model.info())
results = model.train(**{"cfg":"/data/seekyou/Algos/ultralytics/ultralytics/cfg/default.yaml"})
# 

# results = model.train(resume=True)