import numpy as np
from ultralytics import YOLO

model = YOLO('mancha.pt')
model('VRs/', iou=0.2, conf=0.2) 
model = model.numpy()
print(np.shape(model))
