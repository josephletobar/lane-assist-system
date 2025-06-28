import matplotlib.pylab as plt
import cv2
import numpy as np
from cv_utils.PerspectiveTransformation import PerspectiveTransformation
from cv_utils.Thresholding import Thresholding

# global variables for line tracking
prev_left = None
prev_right = None

def filter_outliers(lines, z_thresh=2.0):
    if len(lines) < 2:
        return lines

    # Convert lines to slope and intercept form
    line_data = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:  # avoid vertical lines
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        line_data.append((x1, y1, x2, y2, slope, intercept))

    if len(line_data) < 2:
        return [l[:4] for l in line_data]

    # Calculate slope statistics
    slopes = np.array([d[4] for d in line_data])
    mean = np.mean(slopes)
    std = np.std(slopes)

    # Keep lines within z-score threshold
    filtered = [d[:4] for d in line_data if abs((d[4] - mean) / std) <= z_thresh]
    return filtered


def draw_the_lines(img, lines):
    pf = PerspectiveTransformation()
    img = np.copy(img)  # copy the image
    blank_image = np.zeros(img.shape, dtype=np.uint8)  # create a black canvas
    
    y_bottom = img.shape[0]          # bottom of the image
    y_top = int(img.shape[0] * 0.5)  # top point for drawing

    low_confidence = False

    # Variables to store lane information
    smoothed_left = None
    smoothed_right = None

    if len(lines) < 10:
        low_confidence = True

        cv2.putText(
            img,
            "No Lanes Found",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
            thickness=2)
    
    if lines is not None or len(lines) < 10:
        left_lines = []
        right_lines = []

        mid_x = img.shape[1] // 2  # center of the image
        
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.6:
                continue  # skip horizontal or almost flat lines

    
            # Use position to determine left/right
            avg_x = (x1 + x2) // 2
            
            # if a left line
            if avg_x < mid_x:
                left_lines.append((x1, y1, x2, y2))

                cv2.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

            # if a right line
            else:
                right_lines.append((x1, y1, x2, y2))

                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        # TODO: edit to average by slope and intercept instead of endpoints

        # DRAW averaged left line
        global prev_left
        left_lines = filter_outliers(left_lines)
        if left_lines:
            x1s = [x1 for x1, y1, x2, y2 in left_lines]
            y1s = [y1 for x1, y1, x2, y2 in left_lines]
            x2s = [x2 for x1, y1, x2, y2 in left_lines]
            y2s = [y2 for x1, y1, x2, y2 in left_lines]

            avg_left = (
                int(np.mean(x1s)),
                int(np.mean(y1s)),
                int(np.mean(x2s)),
                int(np.mean(y2s))
            )
            curr_left = np.array(avg_left, dtype=np.float32)

            # apply smoothing
            if prev_left is None:
                smoothed_left = curr_left
            else:
                smoothed_left = 0.9 * prev_left + 0.1 * curr_left
            prev_left = smoothed_left

            x1, y1, x2, y2 = smoothed_left
            
            slope = (y2 - y1) / (x2 - x1)
            x_top = (y_top - y1) / slope + x1
            x_bottom = (y_bottom - y1) / slope + x1

            cv2.line(blank_image, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), (0, 255, 0), 15)

        # Use previous if no new lines detected
        elif prev_left is not None:
            x1, y1, x2, y2 = prev_left
            smoothed_left = prev_left
            
            slope = (y2 - y1) / (x2 - x1)
            x_top = (y_top - y1) / slope + x1
            x_bottom = (y_bottom - y1) / slope + x1

            # cv2.line(blank_image, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), (0, 255, 0), 15)

        # DRAW averaged right line
        global prev_right
        right_lines = filter_outliers(right_lines)
        if right_lines:
            x1s = [x1 for x1, y1, x2, y2 in right_lines]
            y1s = [y1 for x1, y1, x2, y2 in right_lines]
            x2s = [x2 for x1, y1, x2, y2 in right_lines]
            y2s = [y2 for x1, y1, x2, y2 in right_lines]

            avg_right = (
                int(np.mean(x1s)),
                int(np.mean(y1s)),
                int(np.mean(x2s)),
                int(np.mean(y2s))
            )
            curr_right = np.array(avg_right, dtype=np.float32)

            # apply smoothing
            if prev_right is None:
                smoothed_right = curr_right
            else:
                smoothed_right = 0.9 * prev_right + 0.1 * curr_right
            prev_right = smoothed_right

            x1, y1, x2, y2 = smoothed_right

            slope = (y2 - y1) / (x2 - x1)
            x_top = (y_top - y1) / slope + x1
            x_bottom = (y_bottom - y1) / slope + x1

            cv2.line(blank_image, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), (0, 255, 0), 15)

        # Use previous if no new lines detected
        elif prev_right is not None:
            x1, y1, x2, y2 = prev_right
            smoothed_right = prev_right
            
            slope = (y2 - y1) / (x2 - x1)
            x_top = (y_top - y1) / slope + x1
            x_bottom = (y_bottom - y1) / slope + x1

            # cv2.line(blank_image, (int(x_top), int(y_top)), (int(x_bottom), int(y_bottom)), (0, 255, 0), 15)

    # Calculate lane position and offset
    offset_meters = 0
    car_center_x = img.shape[1] / 2

    left_mid_x = (smoothed_left[0] + smoothed_left[2]) / 2
    right_mid_x = (smoothed_right[0] + smoothed_right[2]) / 2
    lane_center_x = (left_mid_x + right_mid_x) / 2
    
    lane_width_pixels = abs(right_mid_x - left_mid_x)
    xm_per_pix = 3.7 / lane_width_pixels if lane_width_pixels != 0 else 0
    
    offset_pixels = car_center_x - lane_center_x
    offset_meters = offset_pixels * xm_per_pix

    if abs(offset_meters) > 0.5: 
        if not low_confidence:
            cv2.putText(
                img,
                "Bad Lane Keeping",
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2)
    else:
        if not low_confidence:
            cv2.putText(
                img,
                "Good Lane Keeping",
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2)

    cv2.putText(
        img,
        "Vehicle is {:.2f} m away from center".format(abs(offset_meters)) if not low_confidence else "Insufficient lane data",
        org=(10, 100),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.66,
        color=(255, 255, 255),
        thickness=2)

    # Add lines to image using perspective transformation
    img = cv2.addWeighted(img, 0.8, pf.backward(blank_image), 1, 0.0)

    return img

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

# For video processing
test_video = "test1"
cap = cv2.VideoCapture(f'data/{test_video}.mp4')

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