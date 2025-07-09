import matplotlib.pylab as plt
import cv2
import numpy as np
from ml_utils.deeplab.deeplab_predict import deeplab_predict
from ml_utils.unet.unet_predict import unet_predict


# For video processing
test_video = "test1"
cap = cv2.VideoCapture(f'test_videos/{test_video}.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    _, pred_mask = deeplab_predict(frame) # run model on current frame

    mask_uint8 = (pred_mask * 255).astype(np.uint8)  # convert from 0/1 float to 0-255 uint8

    # Add Gaussian blur
    mask_blurred = cv2.GaussianBlur(mask_uint8, (5, 5), 0)  # 5x5 kernel, 0 sigma (auto)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)) # Define a kernal for closing
    mask_closed = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, kernel) # Apply morphological closing (dillation followed by erosion)

    mask_color = cv2.cvtColor(mask_closed, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels of color

    # multiply mask_color by green tint
    colored_mask = np.zeros_like(mask_color)
    colored_mask[:, :, 1] = mask_color[:, :, 1]  # green channel only

    # Resize mask to match frame size (resize colored_mask, not mask_color)
    colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))

    # blend with frame
    output = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)

    cv2.imshow("Video", output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()