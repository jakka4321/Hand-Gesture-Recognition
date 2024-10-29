import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
os.chdir(r'C:\Users\Ajaya\Downloads\Hand Gesture Recognition\CAPSTONE PROJECT 1')
import operator

path1='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/hello/'
path2='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/hello/'
path3='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/done/'
path4='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/done/'
path5='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/no/'
path6='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/no/'
path7='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/yes/'
path8='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/yes/'
path9='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/thanks/'
path10='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/thanks/'
path11='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/nothing/'
path12='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/nothing/'
path13='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/right/'
path14='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/right/'
path15='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/train/left/'
path16='C:/Users/Ajaya/Downloads/Hand Gesture Recognition/CAPSTONE PROJECT 1/validation/left/'

cap=cv2.VideoCapture(0)
i=0
image_count=0

while i<25:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    
    roi=frame[120:400,320:620]
    roi=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    roi=cv2.GaussianBlur(roi,(5,5),2)
    roi=cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    _,roi=cv2.threshold(roi,120,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('Image after applying Blurring',roi)
    
    copy=frame.copy()
    cv2.rectangle(copy,(320,120),(620,400),(255,0,0),5)
    
    if i==0:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==1:
        image_count+=1
        cv2.putText(copy,'Recording 1st gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path1+str(image_count)+'.jpg',roi)
    if i==2:
        image_count+=1
        cv2.putText(copy,'Recording 1st gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path2+str(image_count)+'.jpg',roi)
    if i==3:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==4:
        image_count+=1
        cv2.putText(copy,'Recording 2nd gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path3+str(image_count)+'.jpg',roi)
    if i==5:
        image_count+=1
        cv2.putText(copy,'Recording 2nd gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path4+str(image_count)+'.jpg',roi)
    if i==6:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==7:
        image_count+=1
        cv2.putText(copy,'Recording 3rd gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path5+str(image_count)+'.jpg',roi)
    if i==8:
        image_count+=1
        cv2.putText(copy,'Recording 3rd gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path6+str(image_count)+'.jpg',roi)
    if i==9:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==10:
        image_count+=1
        cv2.putText(copy,'Recording 4th gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path7+str(image_count)+'.jpg',roi)
    if i==11:
        image_count+=1
        cv2.putText(copy,'Recording 4th gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path8+str(image_count)+'.jpg',roi)
    if i==12:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==13:
        image_count+=1
        cv2.putText(copy,'Recording 5th gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path9+str(image_count)+'.jpg',roi)
    if i==14:
        image_count+=1
        cv2.putText(copy,'Recording 5th gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path10+str(image_count)+'.jpg',roi)
    if i==15:
        image_count+=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==16:
        image_count+=1
        cv2.putText(copy,'Recording Blank Background - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path11+str(image_count)+'.jpg',roi)
    if i==17:
        image_count+=1
        cv2.putText(copy,'Recording Blank Background - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path12+str(image_count)+'.jpg',roi)
    if i==18:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==19:
        image_count+=1
        cv2.putText(copy,'Recording 6th gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path13+str(image_count)+'.jpg',roi)
    if i==20:
        image_count+=1
        cv2.putText(copy,'Recording 6th gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path14+str(image_count)+'.jpg',roi)
    if i==21:
        image_count=0
        cv2.putText(copy,'Hit Enter to record when ready',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    if i==22:
        image_count+=1
        cv2.putText(copy,'Recording 7th gesture - Train',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path15+str(image_count)+'.jpg',roi)
    if i==23:
        image_count+=1
        cv2.putText(copy,'Recording 7th gesture - Validation',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(copy,str(image_count),(400,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imwrite(path16+str(image_count)+'.jpg',roi)
    if i==24:
        cv2.putText(copy,'Hit Enter to Exit',(70,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.imshow('Image Capture',copy)
    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==13:
        image_count=0
        i+=1
    if interrupt & 0xFF==ord('p'):
        cv2.waitKey(-1)
        
cap.release()
cv2.destroyAllWindows()