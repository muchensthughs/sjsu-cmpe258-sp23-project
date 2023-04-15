import shutil
import random
import glob, os

SOURCE_IMAGE_FOLDER = "FlowChart/FCData/images"
SOURCE_ANNOTATION_FOLDER = "FlowChart/FCData/annotation"
DEST_TRAIN_IMAGE_FOLDER = "FlowChart/FlowChart_train"
DEST_TEST_IMAGE_FOLDER = "FlowChart/FlowChart_test"
DEST_ANNOTATION_TRAIN_FOLDER = "FlowChart/FlowChart_annotation/train"
DEST_ANNOTATION_TEST_FOLDER = "FlowChart/FlowChart_annotation/test"


def seperate_train_test(test_ratio):
    for img_file in glob.glob(SOURCE_IMAGE_FOLDER+'/*.png'):
        img_name = os.path.basename(img_file).split(".")[0]
        xml_file = os.path.join(SOURCE_ANNOTATION_FOLDER, img_name + '.xml')
        if random.random() > test_ratio:
            shutil.copy(img_file, DEST_TRAIN_IMAGE_FOLDER)
            shutil.copy(xml_file, DEST_ANNOTATION_TRAIN_FOLDER)
        else:
            shutil.copy(img_file, DEST_TEST_IMAGE_FOLDER)
            shutil.copy(xml_file, DEST_ANNOTATION_TEST_FOLDER)


def empty_folder(folder):
    files = glob.glob(folder+'/*')
    for f in files:
        os.remove(f)

def clear_all_folders():
    empty_folder(DEST_TRAIN_IMAGE_FOLDER)
    empty_folder(DEST_TEST_IMAGE_FOLDER)
    empty_folder(DEST_ANNOTATION_TRAIN_FOLDER)
    empty_folder(DEST_ANNOTATION_TEST_FOLDER)

def re_shuffle_train_test(test_ratio = 0.2):
    clear_all_folders()
    seperate_train_test(test_ratio)


if __name__ == "__main__":
    re_shuffle_train_test()