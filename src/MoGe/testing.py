import cv2
import torch
# from moge.model.v1 import MoGeModel
# from moge.model.v2 import MoGeModel
from moge.model.v3 import MoGeModel # Let's try MoGe-3

device = torch.device("cuda")

# Load the model
model = MoGeModel.from_pretrained("PATH_TO_CKPT.pt").to(device)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("example_images/02_Office.jpg"), cv2.COLOR_BGR2RGB)                       
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

# Infer
# Three refinement steps are applied by default. Set `refine_steps` to change this.
output = model.infer(input_image)
"""
`output` contains the final prediction. Pass `return_per_step=True` to also return every refinement step.
All maps have the same height and width as the input image.
{
  "points": (H, W, 3),                  # final metric point map in OpenCV camera coordinates (x right, y down, z forward)
  "depth": (H, W),                      # final metric depth map
  "intrinsics": (3, 3),                 # normalized camera intrinsics for the final prediction
  "mask": (H, W),                       # binary mask for valid pixels
  "normal": (H, W, 3),                 # normal map in OpenCV camera coordinates (optional)
}
With `return_per_step=True`, `points_per_step`, `depth_per_step`, and `intrinsics_per_step`
contain `refine_steps + 1` entries, including the initial prediction.
"""