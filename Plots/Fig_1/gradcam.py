import os, sys, torch, numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet152
import torch.nn as nn
from captum.attr import LayerGradCam, LayerAttribution
from torchvision.datasets import ImageFolder

# ────────────────── paths & constants ──────────────────
IMG_PATH = sys.argv[1] if len(sys.argv) > 1 else input("Path to image: ").strip()
DATA_ROOT = "/home/reshma/Otolith/otolith_Final/model_data/test"  # ImageFolder-style test dir
WEIGHTS   = ("/home/reshma/Otolith/otolith_Final/src/outputs/20250515_095917/"
             "models/resnet152_no_augment_color_size512_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "cam_vis"; os.makedirs(OUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────

# 🔤 Font settings (sans serif, size 7)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 7
})

# 1️⃣  Get class name list once from the folder structure
class_names = ImageFolder(DATA_ROOT).classes      # same ordering as training
NUM_CLASSES = len(class_names)

# 2️⃣  Build & load model
model = resnet152(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
model.to(DEVICE).eval()

# 3️⃣  Pre-processing
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# 4️⃣  Open image & ground-truth name
orig = Image.open(IMG_PATH).convert("RGB")
gt_name = os.path.basename(os.path.dirname(IMG_PATH))  # folder name is label
img_tensor = transform(orig).unsqueeze(0).to(DEVICE).requires_grad_(True)

# 5️⃣  Grad-CAM
target_layer = model.layer4[-1]
gradcam = LayerGradCam(model, target_layer)

with torch.no_grad():
    logits = model(img_tensor)
pred_idx = logits.argmax(dim=1).item()
pred_name = class_names[pred_idx]

attr = gradcam.attribute(img_tensor, target=pred_idx)
attr = LayerAttribution.interpolate(attr, img_tensor.shape[-2:])
heat = attr.squeeze().detach().cpu().numpy()
heat = np.maximum(heat, 0); heat /= (heat.max() + 1e-8)

# 6️⃣  Denormalise for display
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
vis = img_tensor[0].cpu()*std + mean
vis = torch.clamp(vis, 0, 1)
vis_pil = T.ToPILImage()(vis)

# 7️⃣  Figure + legend
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(vis_pil)
hm = ax.imshow(heat, cmap="jet", alpha=0.45)
ax.set_axis_off()

# Use smaller sans-serif font for title & colorbar
ax.set_title(f"GT ➜ {gt_name}   |   Pred ➜ {pred_name}", fontsize=7)

# cbar = plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("Relative activation", rotation=270, labelpad=10, fontsize=7)

plt.tight_layout()
out_file = os.path.join(OUT_DIR, "gradcam_single_3.png")
plt.savefig(out_file, dpi=1200, bbox_inches="tight", transparent=True)
plt.show()

print(f"Saved Grad-CAM overlay: {out_file}")
print(f"Ground truth label : {gt_name}")
print(f"Predicted label    : {pred_name}")
