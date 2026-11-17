from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = (
    Path(__file__).resolve().parent.parent
    / "sem_map/data/public_datasets/rellis-3d"
)

LABELS_ZIP = ROOT / "Rellis_3D_pylon_camera_node_label_id.zip"


def select_frames(archive_paths, sequence, count):
    prefix = f"Rellis-3D/{sequence}/pylon_camera_node_label_id/"

    frames = sorted(
        Path(path).stem
        for path in archive_paths
        if path.startswith(prefix) and path.endswith(".png")
    )

    indices = np.linspace(
        0, len(frames) - 1, count, dtype=int
    )

    return [frames[i] for i in indices]


with ZipFile(LABELS_ZIP) as archive:
    paths = archive.namelist()

    train_frames = select_frames(paths, "00000", 20)
    val_frames = select_frames(paths, "00001", 5)

print("TRAIN (00000):")
for frame in train_frames:
    print(frame)

print("\nVAL (00001):")
for frame in val_frames:
    print(frame)

from io import BytesIO
from PIL import Image

# Коэффициенты проходимости для Clearpath Warthog.
# Используем те же правила, что и при оценке.
TRAVERSABILITY = {
    1: 1.0,     # dirt
    3: 1.0,     # grass
    4: 0.0,     # tree
    5: 0.0,     # pole
    6: 0.0,     # water
    8: 0.0,     # vehicle
    9: 0.0,     # object
    10: 1.0,    # asphalt
    12: 0.0,    # building
    15: 0.0,    # log
    17: 0.0,    # person
    18: 0.0,    # fence
    19: 0.2,    # bush
    23: 1.0,    # concrete
    27: 0.0,    # barrier
    29: 1.0,    # uphill
    30: 1.0,    # downhill
    31: 0.9,    # puddle
    33: 0.9,    # mud
    34: 0.0,    # rubble
}

IGNORE_IDS = {0, 7}  # void, sky

OUTPUT = ROOT / "warthog_binary"
(OUTPUT / "images").mkdir(parents=True, exist_ok=True)
(OUTPUT / "segmentations").mkdir(parents=True, exist_ok=True)

with (
    ZipFile(ROOT / "Rellis_3D_pylon_camera_node.zip") as rgb_zip,
    ZipFile(ROOT / "Rellis_3D_pylon_camera_node_label_id.zip") as label_zip,
):
    for split, sequence, frames in [
        ("train", "00000", train_frames),
        ("val", "00001", val_frames),
    ]:
        indices = []

        for frame in frames:
            index = f"{len(indices) + (0 if split == 'train' else 20):06d}"

            rgb_path = (
                f"Rellis-3D/{sequence}/pylon_camera_node/{frame}.jpg"
            )
            gt_path = (
                f"Rellis-3D/{sequence}/pylon_camera_node_label_id/{frame}.png"
            )

            # RGB -> PNG
            rgb = Image.open(BytesIO(rgb_zip.read(rgb_path))).convert("RGB")
            rgb.save(OUTPUT / "images" / f"{index}.png")

            # Официальная семантическая маска
            gt = np.array(
                Image.open(BytesIO(label_zip.read(gt_path)))
            )

            unknown = set(np.unique(gt)) - set(TRAVERSABILITY) - IGNORE_IDS
            if unknown:
                raise ValueError(f"Неизвестные классы: {unknown}")

            # По умолчанию 255 — игнорируемые пиксели
            binary = np.full(gt.shape, 255, dtype=np.uint8)

            for class_id, weight in TRAVERSABILITY.items():
                binary[gt == class_id] = int(weight >= 0.5)

            Image.fromarray(binary).save(
                OUTPUT / "segmentations" / f"{index}.png"
            )

            indices.append(index)

        (OUTPUT / f"{split}_images.txt").write_text(
            "\n".join(indices) + "\n"
        )

        print(f"{split}: сохранено {len(indices)} пар")

print("Датасет подготовлен:", OUTPUT)
