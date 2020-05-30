import numpy as np
import operator
import cv2
import sys, os
from keras.models import load_model
from keras.models import model_from_json
import json
from PIL import Image
import pygame

pygame.init()
screen = pygame.display.set_mode((900,900) , pygame.RESIZABLE)

CLIP_X1 = 160
CLIP_Y1 = 140
CLIP_X2 = 400
CLIP_Y2 = 360

with open('model_in_json (1).json' , 'r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_weights (1).h5')

cap = cv2.VideoCapture(0)
# import pdb; pdb.set_trace()
while True:
    _, frame = cap.read()
    frame=cv2.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)

    roi=frame[100:300, 100:300]

    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.dilate(mask,kernel,iterations = 4)

    mask = cv2.GaussianBlur(mask,(5,5),100)

    mask = cv2.resize(mask , (256,256))

    cv2.imshow('mask',mask)

    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break

pygame.quit()
cap.release()
cv2.destroyAllWindows()