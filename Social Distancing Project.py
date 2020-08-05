# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 23:15:25 2020

@author: Ahmed omar
"""


import numpy as np
import threading
import time
from datetime import datetime
import cv2
import math


#######################################################################
#####################  Settings
#######################################################################

vname="video"

#(W, H) = (640, 360)

(W, H) = (960, 540)


#######################################################################


# full directory of the video file
vid_path = "./videos/"+vname+".mp4"

# Lable names
labelsPath = "./coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)

#######################################################################


#weightsPath = "./yolov3.weights"  ## https://pjreddie.com/media/files/yolov3.weights
#configPath = "./yolov3.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg

###### use this for faster processing (caution: slighly lower accuracy) ###########

weightsPath = "./yolov3-tiny.weights"  ## https://pjreddie.com/media/files/yolov3-tiny.weights
configPath = "./yolov3-tiny.cfg"       ## https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg


net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
#net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

#net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#######################################################################

#vs = cv2.VideoCapture(0)  ## USe this if you want to use webcam feed
vs = cv2.VideoCapture(vid_path)

#######################################################################


Red_boxes = []
Green_boxes = []


def get_yolo_objects(my_frame):

    #print("start")
    
    Red_boxes.clear() 
    Green_boxes.clear() 
    
    blob = cv2.dnn.blobFromImage(my_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #blob = cv2.dnn.blobFromImage(my_frame, 1 / 300.0, (416, 416), swapRB=True, crop=False)
    
    net.setInput(blob)
    
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and LABELS[classID] == "person" :
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
                
    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    #idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.1)
    
    status = []
    person_info = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            
            person_info.append([x, y, w, h, cen])
            status.append(0)
            
    
    for i in range(len(person_info)):
        for j in range(i+1,len(person_info)):
            
            x_dist = (person_info[j][0] - person_info[i][0])
            y_dist = (person_info[j][1] - person_info[i][1])
            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
            
#            if(d <= max_dist):
            if(d <= getmax(person_info[j][2],person_info[i][2])):                
                status[i] = 1
                status[j] = 1
    
    
    
    for i in range(len(person_info)):
        
        if status[i] == 1:
            Red_boxes.append(person_info[i])
        else:
            Green_boxes.append(person_info[i])
    
    #print("Total person = "+str(len(status))+"  , Red_boxes length = "+str(len(Red_boxes)))
    #print("finisd")
    
            
    return


#######################################################################
#######################################################################


threads = []

def isTreadAlive():
  for t in threads:
    if t.isAlive():
      return True
  return False


#######################################################################
#######################################################################


def getmax(a,b):
    
    if a==b:
        return a
    elif a > b:
        return a
    else:
        return b
  


#######################################################################
#######################################################################
  
def get_fps(video):
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    
    if int(major_ver)  < 3 :
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = video.get(cv2.CAP_PROP_FPS)
        
    return fps

#######################################################################
#######################################################################
#######################################################################
#######################################################################


(grabbed, frame) = vs.read()

#(real_H, real_W) = frame.shape[:2]

get_yolo_objects(frame)

frame_counter = 1


start_time = datetime.now()

video_fps = get_fps(vs)

out = cv2.VideoWriter('outpy_'+vname+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), video_fps, (W,H))


frames_to_skip = int(video_fps /5)
print("frames_to_skip = "+str(frames_to_skip))




while grabbed:
    
    try:
        
        frame=cv2.resize(frame,(W,H))
        

        
        if frame_counter % frames_to_skip == 0 :
            
            #if not isTreadAlive() and frame_counter % 3 == 0 :
            get_yolo_objects(frame)
            
#            threads.clear()
#            yolo_thread = threading.Thread(target=get_yolo_objects, args=(frame,), daemon=True)
            
#            threads.append(yolo_thread)
#            yolo_thread.start()
#            yolo_thread.join() 
        
        
        
        
        
        for Box in Red_boxes:
#            cv2.rectangle(frame, (int(Box[0] * w_rate), int(Box[1] * h_rate)), (int((Box[0] + Box[2]) * w_rate), int((Box[1] + Box[3]) * h_rate)), (0, 0, 150), 2)
            cv2.rectangle(frame, (Box[0], Box[1]), (Box[0] + Box[2], Box[1] + Box[3]), (0, 0, 150), 2)
            
        
        for Box in Green_boxes:
            cv2.rectangle(frame, (Box[0], Box[1]), (Box[0] + Box[2], Box[1] + Box[3]), (0, 255, 0), 2)
        
        out.write(frame)
        cv2.imshow('Social distancing analyser', frame)
        
        frame_counter = frame_counter+ 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
        ## get the next frame
        (grabbed, frame) = vs.read()
        
    except Exception:
        print("Exception")
        break
    
end_time = datetime.now()
duration = end_time - start_time
duration_in_s = duration.total_seconds()

print("duration_in_s = "+str(duration_in_s))


cv2.waitKey(1000)
out.release()
cv2.destroyAllWindows()
vs.release()  


print("frame_counter = "+str(frame_counter))

print("Processing finished: open"+"op_"+vname)


