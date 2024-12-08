from ultralytics import YOLO
import sys

sys.path.append('/data/seekyou/Algos/ultralytics')
model = YOLO()
model._new("/data/seekyou/Algos/ultralytics/ultralytics/cfg/models/v8/yolov8-obb-CA.yaml",task="detect",verbose=True)
print(model.info())
# results = model.train(**{"cfg":"/data/seekyou/Algos/ultralytics/ultralytics/cfg/default.yaml"})
# 

# results = model.train(resume=True)