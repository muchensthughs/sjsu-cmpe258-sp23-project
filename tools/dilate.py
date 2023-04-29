import glob
import cv2
import numpy as np


IMAGE_TRAIN_FOLDER = "FlowChart/FlowChart_train"
IMAGE_TEST_FOLDER = "FlowChart/FlowChart_test"

def cv_show(img):
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def dilate_images_in_folder(folder):
    files = glob.glob(folder+'/*.png')
    for image_file in files:
        img = cv2.imread(image_file)
        kernel = np.ones((8, 8), dtype=np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        ## below code to temperarily view images
        # cv_show(img)
        # h, w, c = erosion.shape
        # r = 416/max(h, w)
        # resized = cv2.resize(erosion, (416, 416))
        # cv_show(resized)
        cv2.imwrite(image_file, erosion)

def main():
    for i, data in enumerate(['train','test']):
        print("start dilating %s images" %data)
        if data == 'train': 
            image_folder = IMAGE_TRAIN_FOLDER
        else:
            image_folder = IMAGE_TEST_FOLDER
        dilate_images_in_folder(image_folder)

if __name__ == "__main__":
    main()

