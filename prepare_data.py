# prepare_data.py
import os
from pathlib import Path
import yaml
import csv

PROJECT_ROOT = Path(__file__).parent

def write_yaml(path, content):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False)

def make_data_yaml(dataset_name):
    ds = PROJECT_ROOT / "datasets" / dataset_name
    assert ds.exists(), f"{ds} not found"
    # build relative paths (relative to dataset yaml file as typical for yolov8)
    data = {
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images' if (ds / 'test').exists() else 'test/images'
    }
    # Read nc and names if you want to keep Roboflow metadata; here try to deduce from labels
    # For safety we leave user to set nc/names manually if needed. We'll attempt a small heuristic:
    # Count unique classes in labels
    classes = set()
    for split in ['train','valid','test']:
        lbl_dir = ds / split / 'labels'
        if lbl_dir.exists():
            for f in lbl_dir.glob("*.txt"):
                with open(f, 'r', encoding='utf-8') as fh:
                    for line in fh:
                        if line.strip()=='':
                            continue
                        parts = line.strip().split()
                        classes.add(parts[0])
    if classes:
        # make names placeholder 'class0','class1' — user should update if Roboflow had human names
        names = [f"class{int(c)}" for c in sorted(map(int, classes))]
    else:
        names = []

    content = {
        'train': data['train'],
        'val': data['val'],
        'test': data['test'],
        'nc': len(names),
        'names': names
    }
    yaml_path = ds / 'data.yaml'
    write_yaml(yaml_path, content)
    print(f"Wrote {yaml_path}")
    return yaml_path

def make_cleanliness_image_labels():
    """
    Create a CSV mapping image -> image-level label for cleanliness:
    Rule: if labels/*.txt contains at least one annotation with class == 0 -> clean (1)
          else -> dirty (0)
    Save to datasets/cleanliness/image_labels.csv with columns: image_path,label
    """
    ds = PROJECT_ROOT / "datasets" / "cleanliness"
    out_csv = ds / "image_labels.csv"
    rows = []
    for split in ['train','valid','test']:
        images_dir = ds / split / 'images'
        labels_dir = ds / split / 'labels'
        if not images_dir.exists():
            continue
        for img in images_dir.iterdir():
            if not img.is_file(): continue
            stem = img.stem
            lbl = labels_dir / (stem + ".txt")
            is_clean = 0
            if lbl.exists():
                with open(lbl, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip()=='':
                            continue
                        parts = line.strip().split()
                        try:
                            cl = int(float(parts[0]))
                        except:
                            cl = None
                        if cl == 0:
                            is_clean = 1
                            break
            # label: 1 == clean, 0 == dirty
            rows.append([str(img.relative_to(PROJECT_ROOT)), is_clean])
    if rows:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image','label'])
            writer.writerows(rows)
        print(f"Wrote {out_csv} ({len(rows)} rows)")
    else:
        print("No images found for cleanliness to write image_labels.csv")

if __name__ == "__main__":
    # generate data.yaml for both datasets
    for ds in ['cleanliness','damage']:
        make_data_yaml(ds)
    make_cleanliness_image_labels()
