#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, \
                         Load_Yolo_model, detect_video_realtime_mp, \
                         get_relationships
from yolov3.configs import *
from yolov3.yolov4 import Create_Yolo

# Change input image
image_path = "./FlowChart/FlowChart_test/writer000_fc_011.png"
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    if not os.path.exists(image_path) or image_path is None:
        print("Image path not valid, using default")
        image_path = "./FlowChart/FlowChart_test/writer000_fc_011.png"
else:
    print("No given image path, using default")
    
# image_path = input("Enter image path: ")
# if not os.path.exists(image_path):
#     print("Path doesn't exist. Using default")
#     image_path = "./FlowChart/FlowChart_test/writer000_fc_011.png"

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov4_custom_Tiny")
image, original_image, objects = detect_image(yolo, image_path, "out.png", input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

# print("Objects:")
# print(objects) # [(id, x1, y1, x2, y2, class, confidence)...]

text2shape, arrow2shape = get_relationships(objects)

print("Text-shape relationships")
print(text2shape)
print("Arrow-shape relationships")
print(arrow2shape)
