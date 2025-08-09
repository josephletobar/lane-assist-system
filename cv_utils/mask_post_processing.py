import cv2
import numpy as np
from ml_utils.deeplab.deeplab_predict import deeplab_predict

def get_lane_offset(mask, frame): 
    h, w = frame.shape[:2] # get the dimensions of the image format

    # ensure 1-channel binary (0/1)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask = (mask > 0.5).astype(np.uint8)  # make the mask binary
                                          # compare every pixel to 0.5, 
                                          # if its greater the result is true meaning it is part of the lane
                                          # if its less the result is false meaning its not part of the lane
    
    xs = np.where(mask[int(h*0.8), :] > 0)[0] # grab the horizontal row 80% down the image and extract the values>0 (lane)

    if xs.size < 2:  # not enough lane pixels
        return None
    lane_center = (xs[0] + xs[-1]) // 2 # average leftmost and rightmost lane pixel in the row to get lane center
    cam_center  = w // 2 # assume cam is centered
    return lane_center - cam_center  # + right, - left

def lane_assist(mask, frame, treshold_px = 80):
    lane_offset = get_lane_offset(mask, frame)

    if abs(lane_offset) <+ treshold_px: 
        status = "Good Lane Keeping"
    else:    
        status = "Bad Lane Keeping"

    offset = abs(lane_offset) # string for cv2 to display

    return status, offset


def post_processing(mask, frame):

    # Convert mask from single-channel float array to 8-bit 3-channel format
    mask_uint8 = (mask * 255).astype(np.uint8)  # convert from 0/1 float to 0-255 uint8
    mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels of color

    # Multiply mask_color by green tint
    colored_mask = np.zeros_like(mask_color)
    colored_mask[:, :, 1] = mask_color[:, :, 1]  # green channel only

    # Resize mask to match frame size (resize colored_mask, not mask_color)
    colored_mask = cv2.resize(colored_mask, (frame.shape[1], frame.shape[0]))

    # Smooth Borders
    mask_blurred = cv2.GaussianBlur(colored_mask, (121, 121), 0) # Gaussian blur with 121x121 kernel
    _, mask_smoothed = cv2.threshold(mask_blurred, 120, 255, cv2.THRESH_BINARY) # Convert to binary mask: pixels>120 set to 255 (white), 
                                                                               # others to 0 (black) using binary thresdhold

    # Get lane assist information and display it
    status, offset = lane_assist(mask_smoothed, frame) 
    # cv2.putText( 
    #     frame,
    #     str(offset),
    #     org=(10, 80),
    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #     fontScale=1,
    #     color=(255, 255, 255),
    #     thickness=2)
    cv2.putText(
        frame,
        status,
        org=(10, 50),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=2)                                                                      

    # Blend with frame
    output = cv2.addWeighted(frame, 1.0, mask_smoothed, 0.5, 0)

    return output

# Testing
if __name__ == "__main__":
    weights = "road_deeplab_model2"
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