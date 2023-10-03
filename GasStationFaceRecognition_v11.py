# Version 11, Author: Zhiyi Li
# Date: 2023.09.29
# Based on facenet pytorch version.
# Add loyalty information. 
# Create a dictionary to store count number of each category person
# Every 10 mins to output the counting information. 
# Add size constraints 

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import numpy as np
import pandas as pd
import os
import sys
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import math
workers = 0 if os.name == 'nt' else 4

import time
import datetime
from datetime import datetime

# Section 1: Prepare the detection and recognition.
# Check the device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Define MTCNN model.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Define Inception Resnet v1 module.
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define data directory, prepare available database for existing faces
def collate_fn(x):
    return x[0]

# Grab camera's view.
# used to record the time when we processed previous frame and current frame 
prev_frame_time = 0
new_frame_time = 0

cam_port = 0
v_cap = cv2.VideoCapture(cam_port)
# Check if camera opened successfully
if (v_cap.isOpened()): 
    print("Error opening video stream or file")
    width  = int(v_cap.get(3))  # float `width`
    height = int(v_cap.get(4))  # float `height`
    print (width, height)
    view_center_x = int(width / 2)
    view_center_y = int(height / 2)
    out_writter = cv2.VideoWriter('samples/video_0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

num = 0
category_num = 0
path = 'data_v1/test_images2'
dir = os.listdir(path)

start_time = datetime.now()

pre_min = start_time.minute
cur_min = start_time.minute

pre_date_str = ""
cur_date_str = ""

# Dictionary to store counting information
count_Dic = {}
local_file_name = "./samples/timeRecordCountingInformationPer10mins_sample1_v1.txt" 
with open(local_file_name, "w") as f:
        outLine = "Station_id" + "," + "Time" + "," + "category" + "," + "counting" + "\n"
        f.write(outLine)

f.close()

# Section 2. Process frames from camera's view one by one.
while(v_cap.isOpened()):
    # Capture frame and display detected face
    success, frame = v_cap.read()
    if not success:
        continue
    
    new_frame_time = time.time()
    origin_frame = frame
    #------------------------------------------------
    # Check existing dataset to prepare for recognition.  
    aligned = []
    names = []
    dir = os.listdir(path)

    if len(dir) == 0:
        print("Empty directory")
    else:
        print("Not empty directory")

    if len(dir) != 0:
        dataset = datasets.ImageFolder(path)
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

        for x, y in loader:
            # print ("x: ", x)
            # print ("y: ", y)
            x_aligned, prob = mtcnn(x, return_prob=True)
            
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

    print ("len of aligned: ", len(aligned))

    #--------------------------------------------------
    same_person = False
    samedate_save_sameperson = False
    
    test_x_aligned, prob = mtcnn(origin_frame, return_prob=True)
    boxes, probs, points = mtcnn.detect(frame, landmarks=True)
    # sys.exit(0)
    largest_w = 0
    largest_h = 0 
    if test_x_aligned != None:
        print("len of dir: ", len(dir))
        for i, (box, point) in enumerate(zip(boxes, points)):
        # for i, box in enumerate(boxes):
            [x0, y0, x1, y1] = box.tolist()
            largest_w = max(largest_w, abs(x0 - x1))
            largest_h = max(largest_h, abs(y0 - y1))

            box_center_x = x0 + int(abs(x0 - x1) / 2)
            box_center_y = y0 + int(abs(y0 - y1) / 2)


            if len(dir) == 0:	# Corner case, the database is empty, first person, just add that
                count_Dic[category_num] = 1 
 
                # Create new category
                new_path = path +"/" + str(category_num)  
                if not os.path.exists(new_path):
                    os.makedirs(new_path) # Create new path

                    # Save the image there. 
                    new_img_name = new_path + "/1.png"
                    cv2.imwrite(new_img_name, origin_frame)
                
                    category_num += 1

                continue
       
            else: # Not the first person
                # Calculate the similarity score 
                aligned.append(test_x_aligned)
                names.append("current")
        
                stack_aligned = torch.stack(aligned).to(device)
                embeddings = resnet(stack_aligned).detach().cpu()

                dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
                # print(pd.DataFrame(dists, columns=names, index=names))
                df = pd.DataFrame(dists, columns=names, index=names).tail(1)	# Get the last record to compare current person with previous persons

                # Handle to find most matched picture. 
                # threshold_v = 0.7
                threshold_v = 1.0
        
                columns = df.columns.tolist()
                # print (columns)

                # Get the second minimum value which is distance of current camera's picture to most matched.               
                first_Min_v = 100
                second_Min_v = 100
                second_Min_c = ""

                for index, row in df.iterrows():
                    for column in columns:
                        cur_v = float(row[column])
                        if cur_v <= first_Min_v:
                            first_Min_v = cur_v
            
                for index, row in df.iterrows():
                    for column in columns:
                        cur_v = float(row[column])
                        if cur_v > first_Min_v and cur_v < second_Min_v:
                            second_Min_v = cur_v
                            second_Min_c = column

                print (second_Min_c, second_Min_v)  

                # Depends on threshold value
                print (second_Min_v, threshold_v)

                if second_Min_v < threshold_v: # Same person
                    print ("Same person")
                    print (second_Min_v, threshold_v, "Identified category: ", second_Min_c)
                    same_person = True
                
                    # Add time constraints here. 
                    cd = datetime.now()
                    cur_date_str = str(cd.date())
                    print ("pre_date_str: ", pre_date_str, "cur_date_str: ", cur_date_str)
                    if cur_date_str != pre_date_str:
                        if category_num in count_Dic:
                            count_Dic[category_num] += 1
                        else:
                            count_Dic[category_num] = 1
                    # continue
                else: 
                    print ("Different person")
                    print (largest_w, largest_h)
            
                    # box_center_x = x0 + int(abs(x0 - x1) / 2)
                    # box_center_y = y0 + int(abs(y0 - y1) / 2)
                    # view_center_x = int(width / 2)
                    # view_center_y = int(height / 2)                    

                    distance_box_viewer_center = math.dist([box_center_x, box_center_y], [view_center_x, view_center_y])
                    print ("distance_box_viewer_center: ", distance_box_viewer_center)

                    if largest_w >= 80 and largest_h >= 80 and distance_box_viewer_center <= 80:
                    # Only size is big enough and near center of view enough 
                        print ("largest_w: ", largest_w, "largest_h: ", largest_h)
                        category_num += 1    
                    
                        count_Dic[category_num] = 1 # Create a new dictionary category
                    
                        new_path = path +"/" + str(category_num)  
                        print ("new_path: ", new_path)

                        if not os.path.exists(new_path):
                            print ("Create new file directory happen ? ")
                            os.makedirs(new_path) # Create new path

                            # Save the image there. 
                            new_img_name = new_path + "/1.png"
                            cv2.imwrite(new_img_name, origin_frame)
 
 
            # Remove last elements in aligned
            del aligned[-1]  
            del names[-1]  

    # Display detected face
    (w,h,c) = frame.shape
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # boxes, probs, points = mtcnn.detect(frame, landmarks=True)
    frame_draw = Image.fromarray(origin_frame).copy()
    draw = ImageDraw.Draw(frame_draw)

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    pre_date_str = cur_date_str

    # converting the fps into integer
    fps = int(fps)
  
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    print ("fps: ", fps)

    if boxes is not None:
        for i, (box, point) in enumerate(zip(boxes, points)):
            # box.tolist() is [x0, y0, x1, y1]
            if same_person == True:
                draw.rectangle(box.tolist(), width=2, outline ="green")
            else:
                draw.rectangle(box.tolist(), width=2, outline ="red")
            
            [x1, y1, x2, y2] = np.array(box.tolist(), int)
           
            # Make sure coordinates locates inside the image
            # print ([x1, y1, x2, y2])
            if x1 >= w:
                x1 = w - 1
                continue

            if x1 < 0:
                x1 = 0
                continue

            if x2 >= w:
                x2 = w - 1
                continue

            if x2 < 0:
                x2 = 0
                continue

            if y1 >= h:
                y1 = h - 1
                continue

            if y1 < 0:
                y1 = 0
                continue

            if y2 >= h:
                y2 = h - 1
                continue

            if y2 < 0: 
                y2 = 0
                continue

            # Crop images with bounding boxes. 
            cropped_image = origin_frame[y1:y2, x1:x2] 
             
    # Transfer frame_draw to arrays
    frame_draw_Arr = np.array(frame_draw) 
   
    # cv2.imshow("Face detection" , frame_draw_Arr)
    # cv2.imwrite("aligned.png", frame_draw_Arr)

    cv2.imshow("Face detection" , frame_draw_Arr)
    out_writter.write(frame_draw_Arr)

    # Record counting information 
    now_time = datetime.now()
    cur_min = now_time.minute
    
    d = (now_time - start_time).total_seconds()
    print ('second: ', d)

    min = int(d / 60) 
    print ('min: ', min)
    
    stationID = "4083408367007383835"
    #  outLine = "Time" + "," + "category" + "," + "count" + "\n"        
    if min % 10 == 0 and cur_min != pre_min:	# For 10 mins, filter out same min
        line = str(now_time)
        for k in count_Dic.keys():
            count = count_Dic[k]
            line = stationID + "," + str(now_time) + "," + str(k) + "," + str(count) + "\n"
            with open(local_file_name, "a") as f:
                f.write(line)
     
    pre_min = cur_min
    pre_date_str = cur_date_str

    cv2.waitKey(1)

out_writter.close()         
   
        
              