import pytesseract
import numpy as np
import cv2

def testOCR(img=''):
    res = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    print(res)

#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# text_obj (id, x1, y1, x2, y2, classname, confidence)
def detectText(text_obj, img):
    x1 = text_obj[1]
    y1 = text_obj[2]
    x2 = text_obj[3]
    y2 = text_obj[4]
    crop_img = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    input = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # eroded = erode(gray)
    # blur = cv2.medianBlur(eroded,5)
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    res = pytesseract.image_to_string(input, lang='eng', config='--psm 6') 
    print(res)
    return res

