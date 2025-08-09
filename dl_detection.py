import matplotlib.pylab as plt
import cv2
import numpy as np
from ml_utils.deeplab.deeplab_predict import deeplab_predict
from ml_utils.unet.unet_predict import unet_predict
from cv_utils.mask_post_processing import post_processing

# Load weights
weights = "road_deeplab_model2"

# For video processing
test_video = "test1"
cap = cv2.VideoCapture(f'test_videos/{test_video}.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    _, pred_mask = deeplab_predict(frame, weights) # run model on current frame to get its prediction mask
    result = post_processing(pred_mask, frame) 

    cv2.imshow("Video", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()