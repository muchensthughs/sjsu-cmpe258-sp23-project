from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
import os


image_path = "FlowChart/FCData/images/writer000_fc_001.png"


model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)

save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME + "_Tiny")
print(save_directory)
model.load_weights(save_directory)

detect_image(model, image_path, "", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))



