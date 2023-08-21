# import sys


# MAIN FILE


import os
import numpy as np
# from OBD import ObjectRecognition
from deep_sort.person_id_model.generate_person_features import generate_detections, init_encoder
from deep_sort.deep_sort_app import run_deep_sort, DeepSORTConfig
from deep_sort.application_util.visualization import cv2
import numpy as np
locations =[]
#os.chdir('C:\My Files\Spyder\ODB\object-tracking-master')

classlabels = []
file_name = 'coco.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean([127.5, 127.5, 127.5])
model.setInputSwapRB(True)

# model = ObjectRecognition()

encoder = init_encoder()
config = DeepSORTConfig()



def getObjects(img,objects=[]):
    locations =[]
    lis =[]
    names = []
    deleted_indx = []
    bboxe =[]
    classIndex, confidence, bbox = model.detect(img, confThreshold=0.55, nmsThreshold=0.1)
    if len(objects) == 0: objects = classlabels
    if len(bbox) != 0:
        for i in range(len(classIndex)):
            class_indx = int(classIndex[i-1])
            class_name = classlabels[class_indx-1]
            if class_name not in objects:
                deleted_indx.append(i-1)
                #print(class_name)
            else:
                names.append(class_name)                      
        names = np.array(names)
        count = len(names)
        #print(bbox)
        bboxe = np.delete(bbox, deleted_indx, axis=0)
    return bboxe



if __name__ == '__main__':
    cap = cv2.VideoCapture('V5.mp4')

    while(cap):
        ret, frame = cap.read()
        boxes = getObjects(frame,objects=['car'])
        #print(boxes)

        if len(boxes) > 0:
            encoding = generate_detections(encoder, boxes, frame)
            run_deep_sort(frame, encoding, config)
            #img = visualization.buildNetwork
            #cv2.imshow("Frame",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()
