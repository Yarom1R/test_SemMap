import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Пути относительно расположения самого скрипта.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "sem_map/data/public_datasets/rellis-3d/test_rellis-3d"

RGB_DIR = DATA_DIR / "rgb"
GT_DIR = DATA_DIR / "label_id"
model_name = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "dinov3s_sem_seg_trail_v4"
)

PRED_DIR = RGB_DIR / f"inference_{model_name}/traversability_raw"
GT_THRESHOLD = 0.5
PRED_THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
# Коэффициенты проходимости для Clearpath Warthog.
# ID соответствуют официальной онтологии RELLIS-3D.
TRAVERSABILITY = {
    3: 1.0,    # grass
    4: 0.0,    # tree
    5: 0.0,    # pole
    8: 0.0,    # vehicle
    15: 0.0,   # log
    17: 0.0,   # person
    19: 0.2,   # bush
    23: 1.0,   # concrete
    27: 0.0,   # barrier
    31: 0.9,   # puddle
    33: 0.9,   # mud
    34: 0.0,   # rubble
}

# Неизвестные области и небо исключаем из оценки.
IGNORE_IDS = {0, 7}

# LUT: class ID -> traversability.
# NaN означает, что пиксель не участвует в оценке.
lut = np.full(256, np.nan, dtype=np.float32)
for class_id, weight in TRAVERSABILITY.items():
    lut[class_id] = weight

rgb_files = sorted(RGB_DIR.glob("*.jpg"))
pred_files = sorted(PRED_DIR.glob("frame_*.npy"))

assert len(rgb_files) == 20, f"Expected 20 RGB frames, got {len(rgb_files)}"
assert len(pred_files) == len(rgb_files), "RGB/prediction count mismatch"

TP = FP = FN = TN = 0
excluded = 0

fn_by_class = {}

class_names = {
    3: "grass",
    23: "concrete",
    31: "puddle",
    33: "mud",
}

fp_by_class = {}

obstacle_names = {
    4: "tree",
    5: "pole",
    8: "vehicle",
    15: "log",
    17: "person",
    19: "bush",
    27: "barrier",
    34: "rubble",
}

for i, rgb_path in enumerate(rgb_files):
    # 1. Загружаем предсказанную проходимость SemMap.
    expected_pred = PRED_DIR / f"frame_{i:06d}.npy"
    assert expected_pred.exists(), f"Missing prediction: {expected_pred}"

    pred = np.load(expected_pred)

    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite prediction values in {expected_pred}")

    # 2. Находим соответствующую GT-маску по имени RGB-кадра.
    gt_path = GT_DIR / f"{rgb_path.stem}.png"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    # 3. Уменьшаем GT до разрешения предсказания.
    # NEAREST сохраняет исходные ID семантических классов.
    with Image.open(gt_path) as gt_image:
        gt_image = gt_image.resize(
            (pred.shape[1], pred.shape[0]),
            resample=Image.Resampling.NEAREST,
        )
        gt_ids = np.asarray(gt_image)

    # Проверяем, что все классы предусмотрены протоколом.
    unknown = set(np.unique(gt_ids)) - set(TRAVERSABILITY) - IGNORE_IDS
    if unknown:
        raise ValueError(f"Unknown class IDs in {gt_path}: {unknown}")

    # 4. Строим эталонную карту проходимости.
    gt_trav = lut[gt_ids]

    # NaN -> игнорируем пиксель.
    valid = np.isfinite(gt_trav)

    # 5. Бинаризуем обе карты по одному порогу.
    gt_positive = gt_trav >= GT_THRESHOLD
    pred_positive = pred >= PRED_THRESHOLD

    fn_mask = valid & gt_positive & ~pred_positive

    fp_mask = valid & ~gt_positive & pred_positive

    ids, counts = np.unique(
        gt_ids[fp_mask],
        return_counts=True
    )

    for class_id, count in zip(ids, counts):
        class_id = int(class_id)
        fp_by_class[class_id] = fp_by_class.get(class_id, 0) + int(count)

    ids, counts = np.unique(
        gt_ids[fn_mask],
        return_counts=True
    )

    for class_id, count in zip(ids, counts):
        class_id = int(class_id)
        fn_by_class[class_id] = fn_by_class.get(class_id, 0) + int(count)

    # 6. Матрица ошибок.
    TP += np.count_nonzero(valid & gt_positive & pred_positive)
    FP += np.count_nonzero(valid & ~gt_positive & pred_positive)
    FN += np.count_nonzero(valid & gt_positive & ~pred_positive)
    TN += np.count_nonzero(valid & ~gt_positive & ~pred_positive)

    excluded += np.count_nonzero(~valid)

# 7. Агрегированные метрики по всем 20 кадрам.
iou = TP / (TP + FP + FN)
f1 = 2 * TP / (2 * TP + FP + FN)
false_pass_rate = FP / (FP + TN)

print(f"Frames: {len(rgb_files)}")
print(f"GT threshold: {GT_THRESHOLD}")
print(f"Prediction threshold: {PRED_THRESHOLD}")
print(f"Excluded pixels: {excluded}")
print()
print(f"TP = {TP}")
print(f"FP = {FP}")
print(f"FN = {FN}")
print(f"TN = {TN}")
print()
print(f"IoU: {iou:.4f} ({iou * 100:.2f}%)")
print(f"F1:  {f1:.4f} ({f1 * 100:.2f}%)")
print()
print(
    f"False Pass Rate = FP / (FP + TN) = "
    f"{FP} / {FP + TN} = {false_pass_rate:.4f} "
    f"({false_pass_rate * 100:.2f}%)"
)

print("\nFN по классам Ground Truth:")

for class_id, count in sorted(
    fn_by_class.items(),
    key=lambda item: item[1],
    reverse=True,
):
    name = class_names.get(class_id, f"unknown_{class_id}")
    share = count / FN * 100 if FN else 0

    print(f"{name:12s}: {count:10d} ({share:.2f}% от всех FN)")

assert sum(fn_by_class.values()) == FN

print("\nFP по классам Ground Truth:")

for class_id, count in sorted(
    fp_by_class.items(),
    key=lambda item: item[1],
    reverse=True,
):
    name = obstacle_names.get(class_id, f"unknown_{class_id}")
    share = count / FP * 100 if FP else 0

    print(f"{name:12s}: {count:10d} ({share:.2f}% от всех FP)")

assert sum(fp_by_class.values()) == FP
