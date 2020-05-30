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

CLIP_X1 = 100
CLIP_Y1 = 100
CLIP_X2 = 300
CLIP_Y2 = 300

with open('model_in_json (1).json' , 'r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_weights (1).h5')

cap = cv2.VideoCapture(0)
# import pdb; pdb.set_trace()
while True:
    _, FrameImage = cap.read()
    FrameImage =  cv2.flip(FrameImage, 1)
    kernel = np.ones((3,3),np.uint8)
    # cv2.imshow("", FrameImage)
    cv2.rectangle(FrameImage, (CLIP_X1, CLIP_Y1), (CLIP_X2 , CLIP_Y2) , (0,255,0) , 0)

    ROI = FrameImage[CLIP_Y1:CLIP_Y2, CLIP_X1:CLIP_X2]
     

    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)

    ROI = cv2.inRange(ROI, lower_skin, upper_skin)
    
    ROI = cv2.dilate(ROI,kernel,iterations = 4)
    # ROI    = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

    ROI = cv2.GaussianBlur(ROI,(5,5),100)
    

    # contours,hierachy=cv2.findContours(ROI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # cnt = contours[1]
    # cv2.drawContours(FrameImage, contours, -1, (0,255,0), 2)
    cv2.imshow("", FrameImage)
    # ROI = cv2.GaussianBlur(ROI, (5, 5), 0)

    # _, output = cv2.threshold(ROI, 100, 255, cv2.THRESH_BINARY)

    SHOWROI = cv2.resize(ROI, (256, 256))
    # _, output2 = cv2.threshold(SHOWROI, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI", SHOWROI)
    
    ROI = cv2.resize(ROI, (64, 64))

    result = loaded_model.predict(ROI.reshape(1, 64, 64, 1))
    predict =   { 'zero':    result[0][0],
                  'one':    result[0][1],    
                  'two':    result[0][2],
                  'three':    result[0][3],
                  'four':    result[0][4],
                  'five':    result[0][5],
                  'six':    result[0][6],
                  'seven':    result[0][7],
                  'eight':    result[0][8],
                  'nine':    result[0][9],
                  }
    # import pdb; pdb.set_trace()
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    
    # if(predict[0][1] == 1.0):
    print( predict[0][0] )
        # predict_img  = pygame.image.load(os.getcwd() + '/images/' + predict[0][0] + '.jpg')
    # else:
    #     print( ' No Sign ')
        # predict_img  = pygame.image.load(os.getcwd() + '/images/nosign.png')
    # predict_img = pygame.transform.scale(predict_img, (900, 900))
    # screen.blit(predict_img, (0,0))
    # pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()              