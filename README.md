# Lane Assist System

A lane detection system built for deployment in an RV, assisting with real-time lane tracking and departure alerts. It supports both traditional computer vision and deep learning approaches, depending on available compute.
> *Note: This repository contains the lane detection module I developed as part of a larger RV automation project. The full system integrated multiple subsystems (sensors, control logic, UI), but this repo highlights my contributions in computer vision and deep learning for real-time lane tracking.*

- **Traditional CV Mode**  
  Uses OpenCV techniques like Canny edge detection, perspective transforms, and Hough lines for efficient lane detection.

  <p align="center">
    <img src="assets/cv.gif" width="600">
  </p>

- **Deep Learning Mode**  
  Uses U-Net or DeepLab models to generate segmentation masks for more robust lane detection if sufficient power and compute is available.

  <p align="center">
    <img src="assets/dl.gif" width="600">
  </p>

## Tools & Technologies
-	Languages: Python
-	Libraries & Frameworks: OpenCV, PyTorch, NumPy, Matplotlib
-	Models: U-Net, DeepLab
-	Techniques: Canny edge detection, perspective transforms, Hough line detection, semantic segmentation

## Project Structure

- `cv_detection.py` – OpenCV-based lane detection  
- `dl_detection.py` – Deep learning-based segmentation  
- `cv_utils/` – OpenCV-based image processsing
- `ml_utils/` – Deep learning inference and training helpers  
- `test_videos/` – Sample driving clips  
- `output/` – Output visualizations

## Requirements

- Python 3.8+  
- OpenCV  
- PyTorch  
- NumPy  
- Matplotlib  

Install all dependencies:

```
pip install -r requirements.txt
```

## Running

> *This system is tuned for a specific RV camera setup. Other vehicles or cameras may require retraining or calibration for best results.*

Run detection on a video:

**For traditional computer vision mode:**

```
python cv_detection.py --video path/to/video
```

**For deep learning mode:**

```
python dl_detection.py --video path/to/video
```
