import torch
import torch.nn as nn
import torchvision.models.segmentation as models

def get_deeplab_model(num_classes=1, weights_path=None, device="cpu"):
    # Load base model
    model = models.deeplabv3_resnet50(pretrained=False)

    # Replace classifier for binary or multi-class segmentation
    model.classifier[4] = nn.Sequential( # 4 is the final classifier layer
        nn.Conv2d(256, num_classes, kernel_size=1), # Set 1 segmentation class (binary)
        nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1) # Use Sigmoid for binary
    )

    # Load pretrained weights (e.g., Berkeley) if provided
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device) # Store model parameters (state dictionary) to memory 

        # Remove pretrained final classifier weights (specific to original classes) so we can insert our custom classifier.
        # The earlier layers learn general features useful for all tasks,
        # but the final classifier is task-specific, so we replace it to match our classes.
        filtered_dict = {k: v for k, v in state_dict.items() if "classifier.4" not in k}

        # Load the filtered weights (all except final classifier) into the model
        model.load_state_dict(filtered_dict, strict=False)
        print(f"Loaded weights from {weights_path}")

    return model.to(device)