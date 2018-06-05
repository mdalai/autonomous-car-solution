import sys, json, base64
import numpy as np
import cv2
from inference_seg import Segmentor
file = sys.argv[-1]
if file == 'run.py':
  print ("Error loading video")
  quit
def encode2(array):
	retval,buffer = cv2.imencode('.png', array)
	return base64.b64encode(buffer).decode("utf-8")

answer_key = {}
frame = 1
seg = Segmentor()

    
cap = cv2.VideoCapture(file)
while cap.isOpened():
	ret, bgr_frame = cap.read()
	if ret==True:            
		rgb_frame = cv2.cvtColor(bgr_frame,cv2.COLOR_BGR2RGB)
		binary_car_result,binary_road_result = seg.get_segmentation_single(rgb_frame)
		answer_key[frame] = [encode2(binary_car_result), encode2(binary_road_result)]            
		frame+=1

	else:
		break
cap.release() 
# Print output in proper json format
print (json.dumps(answer_key))
