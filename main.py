from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

# Check the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN model.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

cam_port = 0
v_cap = cv2.VideoCapture(cam_port)
# Check if camera opened successfully
if (v_cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(v_cap.isOpened()):
    # Capture frame-by-frame
    success, frame = v_cap.read()
    origin_frame = frame
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not success:
        continue
    
    boxes, probs, points = mtcnn.detect(frame, landmarks=True)
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)

    if points is not None:
        for i, (box, point) in enumerate(zip(boxes, points)):
            # print ("i: ", i)
            # print ("box: ", box)
            # print ("point: ", point)
            draw.rectangle(box.tolist(), width=5)
            # for p in point:
            #     draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
            
    # cv2.imshow("Face detection" , frame_draw)
    frame_draw.show()
    cv2.waitKey(1)
    
       
      