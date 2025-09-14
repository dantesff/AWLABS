import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
import torch.nn as nn
import torch.nn.modules.container as container

# Разрешаем необходимые классы
torch.serialization.add_safe_globals([
    tasks.DetectionModel,
    container.Sequential,
    modules.Conv
])

# ⚡ Патчим torch.load чтобы YOLO грузился как раньше
_old_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False  # ключевой фикс!
    return _old_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train(dataset_name, pretrained="yolov8m.pt", epochs=30, imgsz=640, batch=16, name=None):
    ds_path = PROJECT_ROOT / "datasets" / dataset_name
    data_yaml = ds_path / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found. Проверь путь к датасету.")

    if name is None:
        name = f"{dataset_name}_yolov8m"

    print(f"🚀 Training {dataset_name} using data {data_yaml}, pretrained {pretrained}")

    # Загружаем модель
    model = YOLO(pretrained)

    # Обучение
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name
    )

    # Путь к лучшим весам
    run_dir = PROJECT_ROOT / "runs" / "detect" / name / "weights"
    best = run_dir / "best.pt"

    if best.exists():
        dest = MODELS_DIR / f"{dataset_name}_best.pt"
        dest.write_bytes(best.read_bytes())
        print(f"✅ Saved best model to {dest}")
    else:
        print("⚠️ Training finished but best.pt not found. Проверь папку runs/detect/.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['damage', 'cleanliness', 'both'], default="both")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--pretrained', default="yolov8m.pt", help="Путь к yolov8m.pt")

    args = parser.parse_args()

    if args.dataset in ('damage', 'both'):
        train('damage',
              pretrained=args.pretrained,
              epochs=args.epochs,
              imgsz=args.imgsz,
              batch=args.batch,
              name='damage_yolov8m')

    if args.dataset in ('cleanliness', 'both'):
        train('cleanliness',
              pretrained=args.pretrained,
              epochs=args.epochs,
              imgsz=args.imgsz,
              batch=args.batch,
              name='cleanliness_yolov8m')
