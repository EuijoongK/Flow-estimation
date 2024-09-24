import os
import cv2

input_dir  = '/mnt/c/Users/MICS/Desktop/result/'
input_list = [file for file in os.listdir(input_dir) if file.endswith("png")]
input_list = sorted(input_list, key = lambda x : int(x.split('_')[1].split('.')[0]))

test_img = cv2.imread(input_dir + input_list[0])
height, width, channel = test_img.shape
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 10

out = cv2.VideoWriter('./flow2.avi', fourcc, fps, (height, width))


for file in input_list:
    
    img = cv2.imread(input_dir + file)
    img = cv2.transpose(img)
    out.write(img)
    
out.release()