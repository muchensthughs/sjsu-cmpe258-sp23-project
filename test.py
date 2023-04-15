import cv2
  
# path
path = "/Users/muchen/Desktop/Files/sjsu/cmpe258/project/sjsu-cmpe258-sp23-project/FCinkML/writer1_1.inkml"
  
# Using cv2.imread() method
img = cv2.imread(path)
  
# Displaying the image
cv2.imshow('image', img)