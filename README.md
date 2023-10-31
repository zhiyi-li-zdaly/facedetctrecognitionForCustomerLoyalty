# facedetctrecognitionForCustomerLoyalty

This project is to apply Open Source software to detect and recognize customer's face in convenient store in gas stations. It apply open source model FaceNet in PyTorch version. 
The work is inspired by the work from Github: https://github.com/timesler/facenet-pytorch

Especially from examples/infer.ipynb script

It is a PC version script: GasStationFaceRecognition_official.py
Code summary:
1. Catch view from camera faced to a door in c-store.
2. Perform face detection.
3. Compared detected face with stored person face/catgory in the PC by distance similarity function. 
4. If distance similarity < distance threshold:
        Same person
   else:
        Different person
        # Update person face/category in the PC.
5. Create log files
6. Demonstrate the video in PC screen.
   




