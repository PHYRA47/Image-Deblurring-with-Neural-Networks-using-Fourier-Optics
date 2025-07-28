from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- Load and preprocess images ---
def load_image_as_tensor(path: str):
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img)

blurred_path = "../_instruction/images/UNet.png"
sharp_path = "../_instruction/parrots256.png"

blurred_tensor = load_image_as_tensor(blurred_path)
sharp_tensor = load_image_as_tensor(sharp_path)

# --- Compute absolute difference map ---
diff_tensor = (blurred_tensor - sharp_tensor).abs()
diff_image = diff_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format for visualization

# --- Save the difference image (no border, no axes, same size) ---
height, width = diff_image.shape[:2]
dpi = 100
figsize = (width / dpi, height / dpi)

fig = plt.figure(figsize=figsize, dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(diff_image)
plt.savefig("difference_map.png", dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()
