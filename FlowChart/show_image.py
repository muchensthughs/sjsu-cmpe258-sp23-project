import random
import cv2
import numpy as np
from PIL import Image

ID = random.randint(0, 200)
label_txt = "FlowChart/FlowChart_train.txt"
image_info = open(label_txt).readlines()[ID].split()

image_path = image_info[0]
image = cv2.imread(image_path)
height, width, channels = image.shape
print("h: " + str(height) + " w: " + str(width) + " channels: " + str(channels))
for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image,(int(float(bbox[0])),
                                 int(float(bbox[1]))),
                                (int(float(bbox[2])),
                                 int(float(bbox[3]))), (255,0,0), 1)

image = Image.fromarray(np.uint8(image))
image.show()