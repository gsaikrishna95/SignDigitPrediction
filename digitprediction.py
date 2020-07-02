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

with open('model_in_json.json' , 'r') as f:
    model_json = json.load(f)
loaded_model = model_from_json(model_json)
loaded_model.load_weights('model_weights.h5')

cap = cv2.VideoCapture(0)

categories = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE'}

# import pdb; pdb.set_trace()
while True:
    _, FrameImage = cap.read()
    FrameImage =  cv2.flip(FrameImage, 1)

    x1 = int(0.5*FrameImage.shape[1])
    y1 = 10
    x2 = FrameImage.shape[1]-10
    y2 = int(0.5*FrameImage.shape[1])

    cv2.rectangle(FrameImage, (x1-1, y1-1), (x2+1 , y2+1) , (255,0,0) , 1)

    ROI = FrameImage[y1:y2, x1:x2]
     
    ROI = cv2.resize(ROI, (64, 64)) 
    ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(ROI, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", test_image)

    result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
    predict =   { 'zero':    result[0][0],
                  'one':    result[0][1],    
                  'two':    result[0][2],
                  'three':    result[0][3],
                  'four':    result[0][4],
                  'five':    result[0][5]
                  }
    # import pdb; pdb.set_trace()
    predict = sorted(predict.items(), key=operator.itemgetter(1), reverse=True)
    # print( predict[0][0] )
    cv2.putText(FrameImage, predict[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("", FrameImage)

    # if(predict[0][1] == 1.0):
    #     predict_img  = pygame.image.load(os.getcwd() + '/images/' + predict[0][0] + '.jpg')
    # else:
    #     predict_img  = pygame.image.load(os.getcwd() + '/images/nosign.png')
    # predict_img = pygame.transform.scale(predict_img, (900, 900))
    # screen.blit(predict_img, (0,0))
    # pygame.display.flip()
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord('q'): # esc key
        break
            
pygame.quit()
cap.release()
cv2.destroyAllWindows()              