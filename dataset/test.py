from image_preprocess import *

result = preprocess('/mnt/c/Users/MICS/Desktop/sample/opencv/vtest.avi')
print(type(result))

print(result[0].shape)