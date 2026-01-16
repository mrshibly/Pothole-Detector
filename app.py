import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the trained model
try:
    model = YOLO('best.pt')
except:
    print("Warning: 'best.pt' not found. Downloading yolov8n.pt for demonstration.")
    model = YOLO('yolov8n.pt')

def detect_potholes(image):
    """
    Function to perform inference on an image.
    """
    if image is None:
        return None
    
    # Run inference
    # verbose=False reduces log clutter
    results = model(image, verbose=False)
    
    # Plot results
    # results[0].plot() returns a BGR numpy array
    res_plotted = results[0].plot()
    
    # Convert BGR to RGB for Gradio
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return res_rgb

# CSS to ensure the video isn't mirrored (good for back cameras)
css = """
video { transform: scaleX(1) !important; }
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕳️ Real-Time Pothole Detection System")
    gr.Markdown("Deploy this on Hugging Face Spaces. Switch tabs for different modes.")
    
    with gr.Tabs():
        # TAB 1: Real-Time Stream
        with gr.Tab("📹 Live Pothole Detection"):
            gr.Markdown("**Use this tab for continuous detection (Video Stream)**")
            with gr.Row():
                with gr.Column():
                    # 'sources=["webcam"]' and NO 'mirror_webcam' argument (handled by CSS)
                    stream_input = gr.Image(sources=["webcam"], label="Live Camera Feed", interactive=True)
                with gr.Column():
                    stream_output = gr.Image(label="Live Detection Output")
            
            # Continuous stream event
            stream_input.stream(fn=detect_potholes, inputs=stream_input, outputs=stream_output, show_progress=False)

        # TAB 2: Upload or Capture
        with gr.Tab("📷 Upload / Take Photo"):
            gr.Markdown("**Use this tab to upload an image or take a single snapshot.**")
            with gr.Row():
                with gr.Column():
                    # Sources allow both upload and webcam snapshot
                    static_input = gr.Image(sources=["upload", "webcam"], label="Upload or Snap Photo", type="numpy")
                    detect_btn = gr.Button("Detect Potholes", variant="primary")
                with gr.Column():
                    static_output = gr.Image(label="Processed Image")
            
            # Button click event
            detect_btn.click(fn=detect_potholes, inputs=static_input, outputs=static_output)
            # Automatic detection on upload change (optional, but good UX)
            static_input.change(fn=detect_potholes, inputs=static_input, outputs=static_output)

# Launch
if __name__ == "__main__":
    demo.launch()
