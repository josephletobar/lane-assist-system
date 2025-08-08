# Lane Assist System

A lane detection system built for deployment in an RV, assisting with real-time lane tracking and departure alerts. It supports both traditional computer vision and deep learning approaches, depending on available compute.

- **Traditional CV Mode**  
  Uses OpenCV techniques like Canny edge detection, perspective transforms, and Hough lines for efficient lane detection.

- **Deep Learning Mode**  
  Uses U-Net or DeepLab models to generate segmentation masks for more robust lane detection if sufficient power and compute is available.

## Project Structure

- `cv_detection.py` – OpenCV-based lane detection  
- `dl_detection.py` – Deep learning-based segmentation  
- `train.py` – Model training script  
- `cv_utils/` – Perspective transforms and post-processing  
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

Run detection on a video:

**For traditional computer vision mode:**

```
python cv_detection.py
```

**For deep learning mode:**

```
python dl_detection.py
```

## Training

Train or fine-tune a model:

```
python train.py --model unet --data path/to/dataset
```
