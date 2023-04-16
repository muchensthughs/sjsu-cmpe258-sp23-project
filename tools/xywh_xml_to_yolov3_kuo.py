import xml.etree.ElementTree as ET
import os
import glob

ANNOTATION_DIR = 'FlowChart/FlowChart_annotation/'
CLASS_NAMES_FILE= "FlowChart/FlowChart.names"
ANNOTATION_TRAIN_FILE = "FlowChart/FlowChart_train.txt"
ANNOTATION_TEST_FILE = "FlowChart/FlowChart_test.txt"
IMAGE_TRAIN_FOLDER = "FlowChart/FlowChart_train"
IMAGE_TEST_FOLDER = "FlowChart/FlowChart_test"
is_subfolder = False

class_names = []

def xywh_to_minmax(x, y, w, h):
    xmin = x 
    ymin = y
    xmax = x + w
    ymax = y + h
    return [xmin, ymin, xmax, ymax]

# return extracted info from one xml file
def parseSingleXML(xml_file):
    tree=ET.parse(open(xml_file))
    root = tree.getroot()
    symbols = root.find("symbols")
    res = ""
    for i, obj in enumerate(symbols.iter("symbol")):
        # print(i, obj.attrib["name"])
        class_name = obj.attrib["name"]
        if class_name not in class_names:
            class_names.append(class_name)
        class_id = class_names.index(class_name)
        bbox = obj.find("bounds")
        # print(bbox.attrib)
        [xmin, ymin, xmax, ymax] = xywh_to_minmax(float(bbox.attrib["x"]), float(bbox.attrib["y"]), float(bbox.attrib["width"]), float(bbox.attrib["height"]))
        bbox_info = (str(int(xmin))+','
                      +str(int(ymin))+','
                      +str(int(xmax))+','
                      +str(int(ymax))+','
                      +str(class_id))
        res += ' ' + bbox_info
    return res

def ParseAllXML(xml_file_folder, file, img_folder):
    for xml_file in glob.glob(xml_file_folder+'/*.xml'):
        img_name = os.path.basename(xml_file).split(".")[0]
        # img_file = os.path.join(img_folder, img_name + '.png')
        img_file = img_folder+"/"+img_name+".png"
        bboxes = parseSingleXML(xml_file)
        res = img_file + bboxes
        file.write(res + '\n')


def XML_to_YOLOv3():
    img_folders = [IMAGE_TRAIN_FOLDER, IMAGE_TEST_FOLDER]
    annotation_files = [ANNOTATION_TRAIN_FILE,ANNOTATION_TEST_FILE]
    # extract annotations for test and train, write this info into annotation txt file
    for i, folder in enumerate(['train','test']):
        annotation_path = os.path.join(os.getcwd(),ANNOTATION_DIR,folder)   
        with open(annotation_files[i], "w") as file:
            ParseAllXML(annotation_path, file, img_folders[i])

    # write class names into classname file
    print("Dataset_names:", class_names)
    with open(CLASS_NAMES_FILE, "w") as file:
        for name in class_names:
            file.write(str(name)+'\n')

if __name__ == "__main__":
    XML_to_YOLOv3()
    # obj = parseSingleXML("/Users/muchen/Desktop/Files/sjsu/cmpe258/project/sjsu-cmpe258-sp23-project/" + ANNOTATION_DIR + "train/writer023_fc_021.xml")
    # print(obj)