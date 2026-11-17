import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Загружаем 4 изображения
img1 = mpimg.imread('./image.png')
img2 = mpimg.imread('./image copy.png')

imgs = [img2, img1]
titles = ['До', 'После']
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