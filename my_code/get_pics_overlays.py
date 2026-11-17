import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Загружаем 4 изображения
img1 = mpimg.imread('sem_map/data/videos/forest1_inference_dinov3s_sem_seg_trail_v4/rgb_inference_dinov3s_sem_seg_trail_v4/segmentation_overlay.png')
img2 = mpimg.imread('sem_map/data/videos/forest1_inference_dinov3s_sem_seg_trail_v4/rgb_inference_dinov3s_sem_seg_trail_v4/traversability_overlay.png')
imgs = [img1, img2]
titles = ['segmentation', 'traversability']
# Создаём сетку 2×2
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# Размещаем изображения
for i, ax in enumerate(axes.flatten()):
    ax.imshow(imgs[i])
    ax.set_title(titles[i])

# Убираем оси координат
for ax in axes.flat:
    ax.axis('off')

# Показываем результат
plt.tight_layout()
plt.show()