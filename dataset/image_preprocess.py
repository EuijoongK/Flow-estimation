import cv2
import numpy as np

def preprocess(input_file):
    
    cap = cv2.VideoCapture(input_file)
    concatenated_frames = []
    buf = []
    
    sample_rate = 4
    index = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if index % sample_rate == 1:
            index += 1
            continue
        
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frame = cv2.resize(grayscale_frame, (256, 256))
        grayscale_frame = grayscale_frame.astype(float) / 255.0
        buf.append(grayscale_frame)
        
        if len(buf) == 2:
            concat_result = np.stack((buf[0], buf[1]))
            concatenated_frames.append(concat_result)
            buf.pop(0)
        
        index += 1
        
    cap.release()
    return np.array(concatenated_frames)
