import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from ml_utils.unet.unet_model import UNet 

# Load model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = "road_unet_model" 

model = UNet()  # initialize UNet model
# model = get_deeplab_model(num_classes=1, device=device)  # initialize DeepLab model

model.load_state_dict(torch.load(f"ml_utils/weights/{weights}.pth", map_location=device))
model.to(device)
model.eval()

# Transform 
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
])

# Predict function
def unet_predict(input_data):
    if isinstance(input_data, str):  # filepath
        image = Image.open(input_data).convert("RGB")
    elif isinstance(input_data, np.ndarray):  # numpy array (BGR from OpenCV)
        image = Image.fromarray(input_data[..., ::-1])  # convert BGR to RGB
    else:
        raise TypeError("Input must be a filepath or a numpy array")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(): 
        output = model(input_tensor) # run the transformed image through the model 
                                     # DeepLab returns dict; use ['out']. UNet returns tensor directly.
        pred_mask = (output.squeeze() > 0.5).float().cpu().numpy() # postprocessing

    return image, pred_mask

# Run on sample images 
if __name__ == "__main__":
    test_dir = "data/images/"
    for filename in sorted(os.listdir(test_dir))[:50]:  # preview x predictions
        if not filename.endswith(".png"): continue
        img_path = os.path.join(test_dir, filename)

        img, pred = unet_predict(img_path)

        # Set up a plot
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Input Image")
        plt.imshow(img)

        plt.subplot(1, 2, 2)
        plt.title("Predicted Mask")
        plt.imshow(pred, cmap="gray")

        plt.suptitle(filename)
        plt.show()