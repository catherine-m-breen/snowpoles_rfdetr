import gradio as gr
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegNano

# 1. Load the model ONCE when the app starts
print("Loading model...")
best_model_path = '/Users/cmbreen/code/snowpoles_rfdetr/checkpoint_best_total.pth'
model = RFDETRSegNano(pretrain_weights=best_model_path, resolution=480)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# 2. Define the function that runs when a user uploads an image
def process_image(input_image):
    # Gradio passes the image in as an RGB NumPy array automatically
    detections = model.predict(input_image)
    
    # Annotate
    h, w = input_image.shape[:2]
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
    color_annotator = sv.ColorAnnotator(color=color)
    polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
    
    annotated_image = input_image.copy()
    annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
    annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)
    
    return annotated_image

# 3. Build the web interface
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Snowpole Image"),
    outputs=gr.Image(type="numpy", label="Model Predictions"),
    title="Snowpole Detection Model",
    description="Upload a photo from your site to see how well the RFDETR model detects snowpoles."
)

# 4. Launch it! (share=True creates a public URL)
if __name__ == "__main__":
    demo.launch(share=True)