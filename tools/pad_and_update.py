import os
import glob
import cv2
import numpy as np

ANNOTATION_DIR = 'FlowChart/FlowChart_annotation/'
CLASS_NAMES_FILE= "FlowChart/FlowChart.names"
ANNOTATION_TRAIN_FILE = "FlowChart/FlowChart_train.txt"
ANNOTATION_TEST_FILE = "FlowChart/FlowChart_test.txt"
IMAGE_TRAIN_FOLDER = "FlowChart/FlowChart_train"
IMAGE_TEST_FOLDER = "FlowChart/FlowChart_test"

def pad_image(image_file):
    image = cv2.imread(image_file)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    output_size = max(height, width)
    pad_top = int((output_size - height)/2)
    pad_left = int((output_size - width)/2)
    background = np.full(shape=(output_size, output_size, channels), fill_value = 255, dtype=np.uint8)
    background[pad_top : height + pad_top, pad_left : width + pad_left, :] = image
    preprocessed_img = cv2.resize(background, (416, 416)) 
    # write new image into current file 
    cv2.imwrite(image_file, preprocessed_img)
    resize_ratio = 416/float(output_size)
    return [pad_top, pad_left], resize_ratio

def update_annotation(annotation_file, padding, ratio, image_file):
    with open(annotation_file, "r") as file: 
        # read the file line by line and update corresponding values
        lines = []
        lines = file.readlines()
        for line_idx, line in enumerate(lines):
            elements = line.split(" ")
            if elements[0] == image_file:
                for i, bbox in enumerate(elements):
                    e = bbox.split(",")
                    if len(e) == 5: 
                        e[0] = str(int(ratio * (int(e[0]) + padding[1]))) # xmin
                        e[2] = str(int(ratio * (int(e[2]) + padding[1]))) # xmax
                        e[1] = str(int(ratio * (int(e[1]) + padding[0]))) # ymin
                        e[3] = str(int(ratio * (int(e[3]) + padding[0]))) # ymax
                    bbox = ",".join(e)
                    elements[i] = bbox
            new_line = " ".join(elements)
            lines[line_idx] = new_line

    with open(annotation_file, "w") as file:
        file.write(''.join(lines))

def pad_and_update():
    for i, data in enumerate(['train','test']):
        print("start %s padding images and updating annotations" %data)
        if data == 'train': 
            annotation_file = ANNOTATION_TRAIN_FILE
            image_folder = IMAGE_TRAIN_FOLDER
        else:
            annotation_file = ANNOTATION_TEST_FILE
            image_folder = IMAGE_TEST_FOLDER
        for image_file in glob.glob(image_folder+'/*.png'):
            print("processing image " + image_file)
            padding, ratio = pad_image(image_file)
            update_annotation(annotation_file, padding, ratio, image_file)

if __name__ == "__main__":
    # update_annotation("/Users/muchen/Desktop/Files/sjsu/cmpe258/project/sjsu-cmpe258-sp23-project/" + ANNOTATION_TRAIN_FILE, [0, 0], "FlowChart/FlowChart_train/writer011_fc_014b.png")
    pad_and_update()