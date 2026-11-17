
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Пути к изображениям
root = Path("../sem_map/data/public_datasets/rellis-3d/Rellis-3D/00003")
frame = "frame000000-1581624075_250"

rgb = np.array(Image.open(
    root / "pylon_camera_node" / f"{frame}.jpg"
))

mask = np.array(Image.open(
    root / "pylon_camera_node_label_id" / f"{frame}.png"
))

# ID класса -> (название, цвет)
classes = {
    3: ("grass", "#006600"),
    4: ("tree", "#00ff00"),
    7: ("sky", "#0000ff"),
    17: ("person", "#cc99ff"),
    19: ("bush", "#ff99cc"),
    27: ("barrier", "#2979FF"),
    31: ("puddle", "#86ffef"),
    33: ("mud", "#634222"),
    34: ("rubble", "#6e168a"),
}

# Преобразуем числовую маску в цветную
colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

for class_id, (name, color) in classes.items():
    color_rgb = tuple(
        int(color[i:i+2], 16) for i in (1, 3, 5)
    )
    colored_mask[mask == class_id] = color_rgb

# Создаём график
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].imshow(rgb)
axes[0].set_title("RGB image")
axes[0].axis("off")

axes[1].imshow(colored_mask)
axes[1].set_title("Ground Truth — semantic segmentation")
axes[1].axis("off")

# Подписываем классы на семантической маске
for class_id, (name, _) in classes.items():
    ys, xs = np.where(mask == class_id)

    if len(xs) == 0:
        continue

    # Находим точку класса, ближайшую к его центру
    cx, cy = np.mean(xs), np.mean(ys)
    idx = np.argmin((xs - cx)**2 + (ys - cy)**2)

    axes[1].text(
        xs[idx], ys[idx], name,
        color="white",
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.7, pad=2)
    )

plt.tight_layout()
plt.show()
