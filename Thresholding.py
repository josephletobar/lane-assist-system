import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)
    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    def __init__(self):
        pass

    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0]  # Hue
        l = hls[:, :, 1]  # Lightness
        s = hls[:, :, 2]  # Saturation

        # White lane (bright lightness)
        white_mask = threshold_rel(l, 0.75, 1.0)

        # Yellow lane (broad hue + decent saturation)
        yellow_mask = threshold_abs(h, 10, 50) & threshold_rel(s, 0.2, 1.0)
        return white_mask | yellow_mask