# infer_and_demo.py
import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import numpy as np
import cv2
import os

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"

# paths to weights
damage_w = MODELS_DIR / "damage_best.pt"
clean_w = MODELS_DIR / "cleanliness_best.pt"
default_w = PROJECT_ROOT / "yolov8m.pt"

damage_model_path = str(damage_w) if damage_w.exists() else str(default_w)
clean_model_path = str(clean_w) if clean_w.exists() else str(default_w)

print("Damage model:", damage_model_path)
print("Cleanliness model:", clean_model_path)

damage_model = YOLO(damage_model_path)
clean_model = YOLO(clean_model_path)

# set confidence thresholds
DAMAGE_CONF = 0.25
CLEAN_CONF = 0.35   # threshold to consider image 'clean' if a 'clean' bbox detected above this

# helper to draw boxes on PIL image using results
def draw_detections(image_pil, results, names):
    draw = ImageDraw.Draw(image_pil)
    for res in results:
        boxes = res.boxes
        for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1,y1,x2,y2 = map(float, box)
            label = names[int(cls)] if names else str(int(cls))
            text = f"{label} {conf:.2f}"
            draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
            draw.text((x1, y1-12), text, fill="white")
    return image_pil

def predict_and_display(image):
    # image is PIL
    img_np = np.array(image)[:,:,::-1]  # pil -> bgr
    # Damage inference
    damage_res = damage_model.predict(source=np.array(image), conf=DAMAGE_CONF, imgsz=640, verbose=False)
    # Clean inference
    clean_res = clean_model.predict(source=np.array(image), conf=CLEAN_CONF, imgsz=640, verbose=False)

    # parse damage results
    damages = []
    for r in damage_res:
        names = r.names
        for b in r.boxes:
            cls = int(b.cls[0].item()) if hasattr(b,'cls') else int(b.cls.item())
            conf = float(b.conf[0].item()) if hasattr(b,'conf') else float(b.conf.item())
            xyxy = b.xyxy[0].tolist() if hasattr(b,'xyxy') else b.xyxy.tolist()
            damages.append({
                "label": names[cls] if names else str(cls),
                "conf": conf,
                "xyxy": xyxy
            })
    # parse clean results (we only need to see if any 'clean' detection exists above threshold)
    is_clean = False
    for r in clean_res:
        for b in r.boxes:
            cls = int(b.cls[0].item()) if hasattr(b,'cls') else int(b.cls.item())
            conf = float(b.conf[0].item()) if hasattr(b,'conf') else float(b.conf.item())
            # if the cleanliness dataset had class index 0 = 'clean'
            if cls == 0 and conf >= CLEAN_CONF:
                is_clean = True
                break

    label_text = "Clean" if is_clean else "Dirty"

    # draw boxes on image for damage only
    pil_out = image.copy()
    draw = ImageDraw.Draw(pil_out)
    for d in damages:
        x1,y1,x2,y2 = d['xyxy']
        draw.rectangle([x1,y1,x2,y2], outline="lime", width=3)
        draw.text((x1, y1-12), f"{d['label']} {d['conf']:.2f}", fill="white")

    return pil_out, label_text, damages

# Gradio UI
title = "Auto Health Demo — Damage detection + Cleanliness"
description = "Загрузите фото. Сначала детектор повреждений покажет bbox (broken/dent/scratch). Затем простая эвристика на основе модели cleanliness возвращает Clean/Dirty."

def gradio_interface(img):
    if img is None:
        return None, "No image", []
    pil = Image.fromarray(img)
    out_img, label_text, damages = predict_and_display(pil)
    return np.array(out_img), label_text, str(damages)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Result with damage boxes"),
        gr.Label(num_top_classes=1, label="Cleanliness label (heuristic)"),
        gr.Textbox(label="Raw damage detections")
    ],
    title=title,
    description=description,
    examples=None
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
