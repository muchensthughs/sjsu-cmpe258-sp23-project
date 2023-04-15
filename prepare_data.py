# Randomly seperate train and test data and annotation into test and train folders under path FlowChart/FCData/
# convert xml, xywh format into text.txt and train.txt in xmin,ymin,xmax,ymax,class_label format

from tools.xywh_xml_to_yolov3 import XML_to_YOLOv3
from tools.seperate_train_test import re_seperate_train_test

def main():
    ## uncomment next line if want to reshuffle and train/test set
    # re_shuffle_train_test(0.2)

    ## convert xml annotaion information into ready to use format
    XML_to_YOLOv3()