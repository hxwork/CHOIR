import cv2
import torch
# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel  # Let's try MoGe-2

device = torch.device("cuda")

# Load the model from huggingface hub (or load from local).
model = MoGeModel.from_pretrained("Ruicheng/moge-v2-vitl-normal/model.pt").to(device)

# Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
input_image = cv2.cvtColor(cv2.imread("../../output/hold_GPMF12_ho3d/rgbs/0.png"), cv2.COLOR_BGR2RGB)
input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

# Infer
output = model.infer(input_image)
metric_depth_map = output['depth'].cpu().numpy()

cv2.imwrite("metric_depth_map.png", metric_depth_map * 255)