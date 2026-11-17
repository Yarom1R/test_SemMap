import numpy as np
import matplotlib.pyplot as plt

# путь к confidence-маске
conf_path = "data/public_datasets/rellis-3d/test_rellis-3d/rgb/frame000190-1581624094_250_inference_dinov3s_sem_seg_warthog_v1/confidence/frame_000000.npy"

conf = np.load(conf_path)

print("shape:", conf.shape)
print("min/max:", conf.min(), conf.max())


plt.figure(figsize=(8, 4))
im = plt.imshow(conf, cmap="viridis", vmin=0.0, vmax=1.0)
cbar = plt.colorbar(im)
cbar.set_label("Confidence")



plt.title("Confidence heatmap")
plt.axis("off")
plt.tight_layout()
plt.savefig("confidence_heatmap_with_colorbar.png", dpi=200, bbox_inches="tight")
plt.show()