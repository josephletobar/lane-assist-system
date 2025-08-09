import argparse
import matplotlib.pylab as plt
import cv2
import numpy as np
from cv_utils.PerspectiveTransformation import PerspectiveTransformation
from cv_utils.Thresholding import Thresholding
from cv_utils.draw_lines import draw_the_lines

def process(image):
    # Initialize perspective transformation
    pf = PerspectiveTransformation()
    
    # Apply perspective transformation instead of cropping
    transformed = pf.forward(image)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray_image)
    
    # Apply blur to reduce noise
    blurred = cv2.GaussianBlur(contrast, (15, 15), 0)
    
    # Edge detection
    canny_image = cv2.Canny(blurred, 50, 100)
    
    # Line detection
    lines = cv2.HoughLinesP(
        canny_image,
        rho=1,                     # distance resolution in pixels
        theta=np.pi / 180,         # angular resolution in radians
        threshold=50,              # minimum number of votes
        minLineLength=40,          # minimum length of line
        maxLineGap=100             # maximum allowed gap
    )
    
    # Draw lines on image
    image_with_lines = draw_the_lines(image, lines)
    
    return image_with_lines

# Parse arguments
parser = argparse.ArgumentParser(description="Lane Assist Video Processing")
parser.add_argument("--video", type=str, required=True, help="Path to video file")
args = parser.parse_args()

# Process the video
cap = cv2.VideoCapture(args.video)

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame wasn't read
    
    processed_frame = process(frame)
    cv2.imshow("Video", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit on 'q' key

cap.release()
cv2.destroyAllWindows()