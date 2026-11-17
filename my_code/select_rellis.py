
from pathlib import Path
import shutil

root = Path("../sem_map/data/public_datasets/rellis-3d")
sequence = root / "Rellis-3D" / "00003"

rgb_dir = sequence / "pylon_camera_node"
mask_dir = sequence / "pylon_camera_node_label_id"

output = root / "test_rellis-3d"
(output / "rgb").mkdir(parents=True, exist_ok=True)
(output / "label_id").mkdir(parents=True, exist_ok=True)


def get_timestamp(path):
    # frame000000-1581624075_250.png
    timestamp = path.stem.split("-")[1]
    seconds, milliseconds = timestamp.split("_")
    return int(seconds) + int(milliseconds) / 1000


# Находим маски с соответствующими RGB-кадрами
pairs = []

for mask in mask_dir.glob("*.png"):
    rgb = rgb_dir / (mask.stem + ".jpg")

    if rgb.exists():
        pairs.append((get_timestamp(mask), rgb, mask))

pairs.sort(key=lambda item: item[0])

if len(pairs) < 20:
    raise ValueError("Недостаточно пар RGB + GT")

# 20 равномерных точек по времени
start = pairs[0][0]
end = pairs[-1][0]

targets = [
    start + i * (end - start) / 19
    for i in range(20)
]

# Выбираем ближайшую доступную пару для каждой точки
remaining = pairs.copy()

for i, target in enumerate(targets, start=1):
    selected = min(
        remaining,
        key=lambda pair: abs(pair[0] - target)
    )
    remaining.remove(selected)

    timestamp, rgb, mask = selected

    shutil.copy2(rgb, output / "rgb" / rgb.name)
    shutil.copy2(mask, output / "label_id" / mask.name)

    print(f"{i:02d}: {rgb.name} | t = {timestamp - start:.1f} s")

print("\nВыбрано 20 пар изображений.")
