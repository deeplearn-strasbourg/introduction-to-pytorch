# -----------
#-------------
#Author: [Vinkle Srivastav](http://camma.u-strasbg.fr/people)
# -------------

## Load the libraries
import numpy as np
import os,sys
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import argparse

# PATHS
parser = argparse.ArgumentParser(description='handdigit classification')
parser.add_argument('--model-path', default='pytorch_mnist.pt', type=str, help='model path')
args = parser.parse_args()
MODEL_STATE_DICT = args.model_path
print('Using Model:', MODEL_STATE_DICT)

## ------- settings -------------------------------------
cap = cv2.VideoCapture(0)
USE_GPU = True
kernel = np.ones((2,2), np.uint8) 
thresh = 100
start_xy = ()
end_xy = ()
cropping = False
pause_video = False

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("using device '{}' ".format(device))
## --------------------------------------------

## ----------- load the trained model ------------------------
class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=20*7*7, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(self.conv2_drop(F.relu(self.conv2(x))), 2)
        x = x.view(-1, 20*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
transform  =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
model = MNIST_Net()
if os.path.isfile(MODEL_STATE_DICT):
    _d = torch.load(MODEL_STATE_DICT)
else:
    print('model path is not correct\n')
    sys.exit(0)
model.load_state_dict(_d['model_state_dict'])
model.to(device)
model.eval()  
## -----------------------------------------------------

## ---------- mouse callback function ------------------
def click_and_crop(event, x, y, flags, param):
    global start_xy, end_xy, cropping
    if event == cv2.EVENT_LBUTTONDOWN:
        start_xy = (x, y)
        cropping = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping is True:
            end_xy = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        end_xy = (x, y)
        cropping = False
cv2.namedWindow('click_and_crop')
cv2.setMouseCallback('click_and_crop',click_and_crop)
## ----------------------------------------------------------

## ------------ Main Loop ------------------------------------
while(True):
    # read the frames if video is paused
    if not pause_video:
        ret, img = cap.read()
        draw = img.copy()
    if cropping:
        # draw the rectangle when user is cropping
        draw = img.copy()
        pause_video = True        
        if start_xy and end_xy:
            cv2.rectangle(draw, start_xy, end_xy, (0, 255, 0), 2)
    else:
        # when cropping is finished; extract and process the ROI
        if start_xy and end_xy:
            # get the ROI
            rect =  img[start_xy[1]:end_xy[1], start_xy[0]:end_xy[0]]
            # thresholding
            ret, thresh = cv2.threshold(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY) , 100, 255,cv2.THRESH_BINARY_INV)
            # resize and dilate
            crop_img = cv2.resize(thresh,(int(28),int(28)))
            crop_img = cv2.dilate(crop_img, kernel, iterations=1) 
            # convert to PIL image
            tensor = Image.fromarray(crop_img, mode='L')            
            # apply the transformation
            tensor = transform(tensor)
            # make it NxCxWxH i.e 4 dimensional
            tensor = tensor.unsqueeze(0)
            # forward pass through model
            outputs = model(tensor)
            # get the prediction
            _, pred = torch.max(outputs, 1)
            # show the ROI and prediction
            cv2.imshow('ROI', crop_img)
            cx, cy = int((start_xy[0] + end_xy[0])/2), int((start_xy[1] + end_xy[1])/2)
            cv2.putText(draw, '->' + str(int(pred.item())), 
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0,255,0), 1, cv2.LINE_AA)

            cv2.rectangle(draw, start_xy, end_xy, (0, 255, 0), 2)
            cv2.imshow('click_and_crop', draw)
            cv2.waitKey(0)
            # reset and start again :) 
            pause_video = False
            start_xy, end_xy = (), ()

    cv2.imshow('click_and_crop', draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
## ------------------------------------------------
