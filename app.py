import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the trained model
# IMPORTANT: You must upload your 'best.pt' file to the same directory as this app.py
try:
    model = YOLO('best.pt')
except:
    print("Warning: 'best.pt' not found. Downloading yolov8n.pt for demonstration.")
    model = YOLO('yolov8n.pt')

def detect_potholes(image):
    """
    Function to perform inference on an image (from camera).
    """
    if image is None:
        return None
    
    # Run inference
    results = model(image)
    
    # Plot results
    # results[0].plot() returns a BGR numpy array
    res_plotted = results[0].plot()
    
    # Convert BGR to RGB for Gradio
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return res_rgb

# Create the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🕳️ Real-Time Pothole Detection")
    gr.Markdown("Deploy this on Hugging Face Spaces (CPU Basic is fine).")
    
    with gr.Row():
        with gr.Column():
            camera_input = gr.Image(source="webcam", streaming=True, label="Live Camera Feed")
        with gr.Column():
            output_image = gr.Image(label="Detection Output")

    # Connect the input to the output
    # Using streaming=True in input and running the function
    # Note: For smoother video, we might want to use gr.Interface with live=True 
    # but 'source="webcam"' in Blocks is also good. 
    # Let's use the simpler Interface for maximum compatibility with free tiers.

    # Alternative simple interface:
    # iface = gr.Interface(fn=detect_potholes, inputs="webcam", outputs="image", live=True)

    camera_input.change(fn=detect_potholes, inputs=camera_input, outputs=output_image)

# Launch
if __name__ == "__main__":
    demo.launch()
