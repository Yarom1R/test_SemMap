import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Загружаем 4 изображения
img1 = mpimg.imread('sem_map/data/videos/four_maps_frames/rgb_0000.png')
img2 = mpimg.imread('sem_map/data/videos/four_maps_frames/seg_0000.png')
img3 = mpimg.imread('sem_map/data/videos/four_maps_frames/trav_0000.png')
img4 = mpimg.imread('sem_map/data/videos/four_maps_frames/conf_0000.png')
img5 = mpimg.imread('sem_map/data/videos/forest1_inference_dinov3s_sem_seg_trail_v4/rgb_inference_dinov3s_sem_seg_trail_v4/segmentation_overlay.png')
img6 = mpimg.imread('sem_map/data/videos/forest1_inference_dinov3s_sem_seg_trail_v4/rgb_inference_dinov3s_sem_seg_trail_v4/traversability_overlay.png')

imgs = [img1, img4, img2, img3,  img5, img6]
titles = ['rgb','confidence', 'segmentation', 'traversability',  'segmentation_overlay', 'traversability_overlay']
# Создаём сетку 2×2
fig, axes = plt.subplots(3, 2, figsize=(10, 8))

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