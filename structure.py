from ultralytics import YOLO

model = YOLO("/data/seekyou/Algos/ultralytics/runs/obb/train/weights/best.pt")
print(model.info(detailed=True))
# results = model.train(resume=True)