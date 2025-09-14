import gradio as gr
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

rf = Roboflow(api_key="3xHadLJ7GhY5tMP7zbdE")

project_parts = rf.workspace().project("car-parts-segmentation")
model_parts = project_parts.version(2).model

project_damage = rf.workspace().project("car-damage-detection-ha5mm")
model_damage = project_damage.version(1).model


def detect_cleanliness(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    value = hsv[:, :, 2].mean()
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    texture_variance = np.var(gray)

    is_dirty = False
    dirt_confidence = 0

    if saturation < 50 and value < 100:
        is_dirty = True
        dirt_confidence = 0.7
    elif texture_variance > 1000:
        is_dirty = True
        dirt_confidence = 0.6

    return is_dirty, dirt_confidence


def analyze_car(image):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    try:
        image_np = np.array(image)
        annotated_image = image_np.copy()

        result_damage = model_damage.predict(img_path, confidence=40).json()
        labels_damage = [item["class"] for item in result_damage["predictions"]]
        detections_damage = sv.Detections.from_inference(result_damage)

        mask_annotator = sv.MaskAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=0.5)

        if len(detections_damage.xyxy) > 0:
            annotated_image = mask_annotator.annotate(
                scene=annotated_image,
                detections=detections_damage
            )

            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections_damage,
                labels=labels_damage
            )

        damaged_parts = []
        if len(detections_damage.xyxy) > 0:
            x1, y1, x2, y2 = map(int, detections_damage.xyxy[0])
            cropped_damage = image_np[y1:y2, x1:x2]

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as damage_tmp:
                cv2.imwrite(damage_tmp.name, cropped_damage)
                damage_path = damage_tmp.name

            result_parts = model_parts.predict(damage_path, confidence=15).json()
            damaged_parts = list(set([item["class"] for item in result_parts["predictions"]]))

            os.unlink(damage_path)

        is_dirty, dirt_confidence = detect_cleanliness(image_np)
        cleanliness_state = "Грязный" if is_dirty else "Чистый"

        cv2.putText(annotated_image, f"Чистота: {cleanliness_state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        damage_count = len(detections_damage.xyxy)

        if damage_count > 0 and is_dirty:
            overall_state = "⚠️ Плохое состояние (Повреждения и грязь)"
        elif damage_count > 0:
            overall_state = "⚠️ Поврежден но чистый"
        elif is_dirty:
            overall_state = "⚠️ Грязный но без повреждений"
        else:
            overall_state = "✅ Отличное состояние"

        report = f"""
        🚗 Отчет о состоянии автомобиля:

        📊 Общее состояние: {overall_state}

        🔍 Обнаружение повреждений:
        - Найдено повреждений: {damage_count}
        - Типы повреждений: {', '.join(labels_damage) if labels_damage else 'Нет'}
        - Задетые детали: {', '.join(damaged_parts) if damaged_parts else 'Нет'}

        🧹 Оценка чистоты:
        - Состояние: {cleanliness_state}
        - Уверенность: {dirt_confidence:.2f}
        - Анализ: {'Обнаружена грязь/пыль' if is_dirty else 'Автомобиль чистый'}

        ⚠️ Рекомендации:
        {'• Требуется осмотр повреждений' if damage_count > 0 else '• Повреждений не обнаружено'}
        {'• Автомобиль нуждается в мойке' if is_dirty else '• Автомобиль чистый'}
        {'• Проверьте ' + ', '.join(damaged_parts) + ' для ремонта' if damaged_parts else ''}
        """

        return annotated_image, report

    finally:
        os.unlink(img_path)


css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: white;
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.column {
    background: white;
    padding: 20px;
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
button {
    background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 15px 30px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
button:hover {
    background: linear-gradient(45deg, #FF8E53, #FF6B6B) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
"""

with gr.Blocks(title="Анализатор состояния автомобиля", css=css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_classes="header"):
        gr.Markdown("# 🚗 Анализатор состояния автомобиля")
        gr.Markdown("Загрузите фото автомобиля для анализа повреждений и чистоты")

    with gr.Row():
        with gr.Column(elem_classes="column"):
            image_input = gr.Image(type="pil", label="📸 Загрузите фото автомобиля")
            analyze_btn = gr.Button("🔍 Проанализировать автомобиль", variant="primary")

        with gr.Column(elem_classes="column"):
            image_output = gr.Image(label="📊 Результаты анализа")
            text_output = gr.Textbox(label="📝 Детальный отчет", lines=18)

    analyze_btn.click(
        fn=analyze_car,
        inputs=image_input,
        outputs=[image_output, text_output]
    )

    with gr.Column(elem_classes="column"):
        gr.Markdown("""
        ### 📋 Что анализируем:
        - **Повреждения**: Царапины, вмятины, повреждения кузова
        - **Чистоту**: Грязь, пыль, загрязнения  
        - **Детали**: Конкретные поврежденные элементы
        - **Общее состояние**: Полная оценка автомобиля
        """)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)