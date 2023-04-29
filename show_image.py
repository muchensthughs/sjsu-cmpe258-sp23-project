import random
import cv2
import numpy as np
from PIL import Image
from yolov3.yolov4 import read_class_names
from yolov3.configs import *

ID = random.randint(0, 200)
label_txt = "FlowChart/FlowChart_train.txt"
image_info = open(label_txt).readlines()[ID].split()

image_path = image_info[0]
image = cv2.imread(image_path)
image_h, image_w, _ = image.shape
bbox_thick = int(0.6 * (image_h + image_w) / 1000)
if bbox_thick < 1: 
    bbox_thick = 1
fontScale = 0.75 * bbox_thick
CLASSES=YOLO_COCO_CLASSES
NUM_CLASS = read_class_names(TRAIN_CLASSES)

for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image,(int(float(bbox[0])),
                                 int(float(bbox[1]))),
                                (int(float(bbox[2])),
                                 int(float(bbox[3]))), (0,255,0), 2)
    label = NUM_CLASS[int(bbox[4])]
    x1 = int(float(bbox[0]))
    y1 = int(float(bbox[1]))
    # get text size
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                            fontScale, thickness=bbox_thick)
    # put filled text rectangle
    cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), (0,128,128), thickness=cv2.FILLED)

    # put text above rectangle
    cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                fontScale, (0,0,128), bbox_thick, lineType=cv2.LINE_AA)

image = Image.fromarray(np.uint8(image))
image.show()