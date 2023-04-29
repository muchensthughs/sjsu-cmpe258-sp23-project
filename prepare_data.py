# Randomly seperate train and test data and annotation into test and train folders under path FlowChart/FCData/
# convert xml, xywh format into text.txt and train.txt in xmin,ymin,xmax,ymax,class_label format

from tools.xywh_xml_to_yolov3 import XML_to_YOLOv3
from tools.seperate_train_test import re_shuffle_train_test
from tools.pad_and_update import pad_and_update

def main():
    ## uncomment next line if want to reshuffle the train/test set
    re_shuffle_train_test(0.2)

    ## convert xml annotaions in annotation folder into ready to use format
    XML_to_YOLOv3()

    ## pad images to square and updates its bbox locations
    pad_and_update()

if __name__ == "__main__":
    main()