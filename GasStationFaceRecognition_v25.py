# Version 24
# Author: Zhiyi Li
# Date: 2023.10.19
# Based on facenet pytorch version.
# Add loyalty information. 
# Create a dictionary to store count number of each category person
# Every 10 mins to output the counting information. 
# Add size constraints 
# Add category, detection, recognization information
# Try to remove duplicates with distance constraints
# Add log file to save information
# Add blur/unfocused detection and shape information
# Modify the counting dictionary to add date information
# Correct counting information.
# Add hash memeory into the database faces
# Apply np.argmin to find minimum index for distance. Handle multiple face in the same image. 
# Version 23: Add counting information, every 10 mins to record the same day person show up  

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms 

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

now_time = datetime.now()

# Section 1: Prepare detection and recognition models.
 
# Check the device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# MTCNN model for face detection.
mtcnn = MTCNN(min_face_size=40, keep_all=True, 
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# Inception Resnet v1 module for face recognition. 
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define data directory, prepare available database for existing faces classes
def collate_fn(x):
    return x[0]

# Grab camera's view.
camera_id = "admin"
camera_password = "AdminAdmin1"
IP_address = "192.168.1.134"
source = "rtsp://" + camera_id + ":" + camera_password + "@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=1"
print (source)

# Get the size of source
v_cap = cv2.VideoCapture(source) 

# Check if camera is opened successfully
if (v_cap.isOpened()): 
    print("Error opening video stream or file")
    width  = int(v_cap.get(3))  # float `width`
    height = int(v_cap.get(4))  # float `height`
    print (width, height)
    view_center_x = int(width / 2)
    view_center_y = int(height / 2)
    out_writter = cv2.VideoWriter('samples/video_25.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (width, height))

import datetime
from datetime import datetime
start_time = datetime.now()
pre_min = start_time.minute
cur_min = start_time.minute

# Record counting information. 
local_file_name = "./samples/timeRecordCountingInformationPer10mins_sample1_v25.txt" 
with open(local_file_name, "w") as f:
    outLine = "Station_id" + "," + "timestamp" + "," + "date" + "," + "category" + "\n"
    f.write(outLine)
f.close()

#
 
# Initialize parameters
num = 0
category_num = 0
recog_category_num = 0

cur_date_str = str(now_time.date())
pre_date_str = cur_date_str

# For log file. 
log_file_name = "./logs/log" + cur_date_str + ".txt"
log_file = open(log_file_name, "a")

path = 'data_v1/test_images25'
face_path = 'data_v1/test_face_images25'

dir = os.listdir(path)
distance_threshold = min(width, height) / 3
front_face = False

count_Dic = {} # A dictionary to store person category information
               # Key: a string of date， for example 2023-10-19.
               # Value: a set of person(category) to show up the same date

# Check existing dataset to prepare for recognition.  
aligned = []
names = []
dir = os.listdir(path)
print (len(dir))

if len(dir) != 0:
    dataset = datasets.ImageFolder(path)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    for x, y in loader:
        faces, prob = mtcnn(x, return_prob=True)
        if faces is not None:
            for face in faces:
                aligned.append(face)
                names.append(dataset.idx_to_class[y])
                  
print("len of aligned: ", len(aligned))

pre_aligned = aligned
pre_names = names

# Section 2. Process frames from camera's view one by one.
while(v_cap.isOpened()):

    # Capture frame and detected face
    success, frame = v_cap.read() 
    
    if not success:
        continue

    num += 1
    if num % 5 == 1: # Every 5 frames, perform detection. 
        print ("num: ", num)
        
        # Inverse the frame in 180 degree
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        original_frame = frame

        # Detect faces       
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
        faces, prob = mtcnn(frame, return_prob=True)
        boxes, probs, points = mtcnn.detect(frame, landmarks=True)

        # verify face
        # Check existing dataset to prepare for recognition.  
        aligned = pre_aligned
        names = pre_names

        dir = os.listdir(path)
        print (len(dir))

        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        
        if boxes is not None:	# Handle multiple faces
            for i, (face, box, point) in enumerate(zip(faces, boxes, points)):
                # Initialization.
                same_person = False
                near_center = False
                front_face = False
                no_tilt_face = False
                no_bow_face = False
                first_person = False

                [x0, y0, x1, y1] = np.array(box.tolist(), int)
                if x0 >= width:
                    x0 = width - 1
                
                if x0 < 0:
                    x0 = 0
                
                if x1 >= width:
                    x1 = width - 1
              
                if x1 < 0:
                    x1 = 0
              
                if y0 >= height:
                    y0 = height - 1
              
                if y0 < 0:
                    y0 = 0
                
                if y1 >= height:
                    y1 = height - 1
              
                if y1 < 0: 
                    y1 = 0
             
                # Crop images with bounding boxes. 
                cropped_image = original_frame[y0:y1, x0:x1]
 
                box_center_x = x0 + int(abs(x0 - x1) / 2)
                box_center_y = y0 + int(abs(y0 - y1) / 2)
                
                box_width =  abs(x0 - x1) 
                box_height = abs(y0 - y1)
      
                # Constraints for face position, assume it nears to center of view, and face is symmetric, not tilted. 
                distance_box_viewer_center = math.dist([box_center_x, box_center_y], [view_center_x, view_center_y])   
                
                # Landmark point 0 to 4 refer to left eye, right eye, nose, left mouth border, right mout boarder
                left_eye_x = point[0][0] # left eye x1
                right_eye_x = point[1][0] # right eye x2
                nose_x = point[2][0] # nose x3
                
                left_eye_y = point[0][1] # left eye y1
                right_eye_y = point[1][1] # right eye y2
                nose_y = point[2][1] # nose y3
                 
                left_mouth_x = point[3][0]
                right_mouth_x = point[4][0]
                left_mouth_y = point[3][1] # left mouth y4
                right_mouth_y = point[4][1] # right mouth y5
              
                if nose_x > (left_eye_x + 2) and nose_x < (right_eye_x - 2) and nose_x > (left_mouth_x + 2) and nose_x < (right_mouth_x - 2): 
                    front_face = True
 
                log_file.write("nose_x: " + str(nose_x) + "left_eye_x: " +  str(left_eye_x) + "right_eye_x: " + str(right_eye_x) + "\n")

                if abs(left_eye_y - right_eye_y) / box_height < 1 / 10:
                    no_tilt_face = True 

                # Apply ratio between nose to mouth to decide whether it is a bow face or not. 
                if abs(nose_y - left_mouth_y) / box_height  >  1 / 5 and abs(nose_y - left_mouth_y) / box_height  >  1 / 5: 
                    no_bow_face = True

                # Add verfication here. Skip not candidate face
                # print ("distance_box_viewer_center: ",  distance_box_viewer_center,  "distance_threshold: ",  distance_threshold)
                log_file.write("distance_box_viewer_center: " + str(distance_box_viewer_center) + "distance_threshold: " +  str(distance_threshold) + "\n")

                if distance_box_viewer_center < distance_threshold:
                    near_center = True

                print ("near_center: ", near_center, "front_face: ", front_face, "no_tilt_face：" ,  no_tilt_face)
                log_file.write("near_center: " + str(near_center) + " front_face: " + str(front_face) + "\n")

                if near_center == False or front_face == False or no_tilt_face == False or no_bow_face == False:
                    continue
                
                # Get time information
                import datetime
                from datetime import datetime
                cur_time = datetime.now() 
                cur_date_str = str(cur_time.date())
                if cur_date_str not in count_Dic:   # Set to store person/category
                    count_Dic[cur_date_str] = set()
                       
                
                # Get previous aligned information
                aligned = pre_aligned

                if len(dir) == 0:	# corner case, the database is empty, first person, just add into the system. 
                    # Enter the person face into the database system.
                    first_person = True

                    # Create new category
                    new_path = path +"/" + str(category_num)  
                    if not os.path.exists(new_path):
                        os.makedirs(new_path) # Create new path

                        # Save the image there. 
                        new_img_name = new_path + "/1.png"
                        cv2.imwrite(new_img_name, original_frame)

                    new_face_path = face_path +"/" + str(category_num)  
                    if not os.path.exists(new_face_path):
                        os.makedirs(new_face_path) # Create new path

                        # Save the cropped image there. 
                        new_face_img_name = new_face_path + "/1.png"
                        cv2.imwrite(new_face_img_name, cropped_image)

                    if box is not None:	# Add face into existing aligned.
                        aligned.append(face)
                        print ("face.shape: ", face.shape)
                        names.append(str(category_num))  

                    if category_num not in count_Dic[cur_date_str]:	# Add the category into dictionary to store cur_date_str person/category.
                        count_Dic[cur_date_str].add(category_num)
                
                else: # Not the first person
                    # Calculate the similarity score
                    print ("Not the first person")
                    first_person = False

                    if box is not None:
                        aligned.append(face)
                        print ("face.shape: ", face.shape)
                        names.append("current")
                    
                    stack_aligned = torch.stack(aligned).to(device)
                    embeddings = resnet(stack_aligned).detach().cpu()
                    print ("embeddings: ", embeddings)
    
                    dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
                    print(pd.DataFrame(dists, columns=names, index=names))
                    
                    print ("Test")

                    df = pd.DataFrame(dists, columns=names, index=names).tail(1) # Get last row
                    print ("df: ", df)
                    df.pop(df.columns[-1])
 
                    results = np.argmin(df, axis=1)                    
                    print("results: ",  results)
                    print("len(results): ",  len(results))
                    min_idx = results[0]
                 
                    # Handle to find most matched picture.    
                    # threshold_v = 0.8
                    threshold_v = 1.0

                    columns = df.columns.tolist()
                    
                    print("columns: ", columns)
                    print("Minimum column: ", columns[results[0]])
                    min_idx = results[0]
                    min_c = columns[results[0]]
                    min_v = float(df.iloc[0, min_idx])
                    
                    print("min_idx: ", min_idx)
                    print("min_c: ", min_c)
                    print("min_v: ",  min_v)   
            
                    # Depends on threshold value
                    if min_v < threshold_v: # Same person

                        print("front_face: ", front_face)
                        print("no_tilt_face: ",no_tilt_face)
                        print("no_bow_face: ", no_bow_face)

                        same_person = True
                        print ( "Same person: ", str(same_person) )
 
                        print (min_v, threshold_v, "Identified category: ", min_c)
                        recog_category_num = int(min_c)
                        print ("recog_category_num: ", recog_category_num)

                        if recog_category_num not in count_Dic[cur_date_str]:	# Add the category into dictionary to store cur_date_str person/category.
                            count_Dic[cur_date_str].add(recog_category_num)
                    
                        # Pop out current for same person
                        if same_person == True: 
                            aligned.pop() 
                            names.pop()
                        
                    else: # Different person
                        same_person = False
                        print ( "Different person: ", str(same_person))
                             
                        print ("distance_box_viewer_center: ", distance_box_viewer_center)
                        print ("distance_threshold: ", distance_threshold)
                        print ("front_face: ", front_face)
                        category_num += 1
                        print ("category_num: ", category_num)
                           
                        # Create new category
                        new_path = path +"/" + str(category_num)  
                        if not os.path.exists(new_path):
                            os.makedirs(new_path) # Create new path

                            # Save the image there. 
                            new_img_name = new_path + "/1.png"
                            cv2.imwrite(new_img_name, original_frame)

                        new_face_path = face_path +"/" + str(category_num)  
                        if not os.path.exists(new_face_path):
                            os.makedirs(new_face_path) # Create new path

                            # Save the cropped image. 
                            new_face_img_name = new_face_path + "/1.png"
                            cv2.imwrite(new_face_img_name, cropped_image)

                        if category_num not in count_Dic[cur_date_str]:	# Add the category into dictionary to store cur_date_str person/category.
                            count_Dic[cur_date_str].add(category_num)
                        
                        if same_person == False:
                            names.pop()
                            names.append(str(category_num))                          
                
                ##########################################################################
                if same_person == True:
                    draw.rectangle(box.tolist(), width=2, outline ="green")
                    font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
                    text = "Recognized \n Category: " + str(recog_category_num)
                    draw.text((x1 + 10, y0 - 10), text, fill ="green", font = font)    
                else:
                    draw.rectangle(box.tolist(), width=2, outline ="red")
                    font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)
                    draw.text((x1 + 10, y0 - 10), "Detected", fill="red", font = font)                

                pre_aligned = aligned # aligned for save       
                pre_names = names

        frame_draw_Arr = np.array(frame_draw) 
        cv2.imshow("Face detection" , frame_draw_Arr)
        out_writter.write(frame_draw_Arr)

        # Record counting information 
        import datetime
        from datetime import datetime

        now_time = datetime.today()
        cur_min = now_time.minute
    
        cur_date_str = str(now_time.date())
        print (cur_date_str)

        d = (now_time - start_time).total_seconds()
        min = int(d / 60) 
    
        stationID = "4083408367007383835"
        if min % 10 == 0 and cur_min != pre_min:	# For 10 mins, filter out same min
            with open(local_file_name, "a") as f:
                for k in count_Dic.keys():
                    cur_date = k
                    categorys = count_Dic[k]
                    for c in categorys:
                        outLine = stationID + "," + str(now_time) + "," + str(cur_date_str) + "," + str(c) + "\n"                       
                        f.write(outLine)

            f.close()

        pre_min = cur_min
        

        cv2.waitKey(1)            
    
out_writter.close()  
log_file.close()       
   
        
              