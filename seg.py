import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import os
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np

# Load pretrained model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Input and output folders
input_dir = "test"
output_dir = "segmented_test"
os.makedirs(output_dir, exist_ok=True)

# Transformation
transform = T.Compose([
    T.Resize((520, 520)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Loop through images
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # Create sheep mask (Pascal VOC class 17 is 'sheep')
    import numpy as np

    sheep_mask = np.isin(output_predictions, [17, 13, 8, 10, 12])# Boolean mask

    # Convert original image to numpy array
    original_resized = image.resize((520, 520))  # match model input size
    image_np = np.array(original_resized)

    # Apply mask: set background to black
    masked_image = image_np.copy()
    masked_image[~sheep_mask] = 0  # background -> black

    # Convert to image and save
    masked_pil = Image.fromarray(masked_image)
    masked_pil.save(os.path.join(output_dir, image_name))



# Load pretrained DeepLabV3 model


# Predict segmentation mask
 # [H, W]

print(output_predictions)

# Visualization: map each class to a color (VOC 21 classes)
VOC_COLORS = np.array([
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # dining table
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # potted plant
    [128, 64, 0],     # sheep ✅
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128]      # tv/monitor
])



# Convert mask to color image
# seg_color = VOC_COLORS[output_predictions]

# # Show segmented image
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")

# plt.subplot(1, 2, 2)
# plt.imshow(seg_color)
# plt.title("Predicted Segmentation")
# plt.show()
