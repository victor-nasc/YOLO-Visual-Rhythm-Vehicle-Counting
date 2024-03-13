import cv2
import numpy as np

cap = cv2.VideoCapture('Set01.mp4')
assert cap.isOpened() == True, 'Can not open the video'

cap.set(cv2.CAP_PROP_POS_FRAMES,2300)


ret, atu = cap.read()
cv2.imshow('frame', atu)
cv2.waitKey(0)