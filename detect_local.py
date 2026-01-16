import cv2
from ultralytics import YOLO

# Load the trained model
# Ensure 'best.pt' is in the same directory
try:
    model = YOLO('best.pt')
except:
    print("Error: 'best.pt' not found. Please ensure your trained model is in this folder.")
    print("Falling back to standard 'yolov8n.pt' for testing...")
    model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set camera resolution (optional, e.g., 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run inference on the frame
    # verbose=False keeps the terminal clean
    results = model(frame, verbose=False)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the resulting frame
    cv2.imshow('Real-Time Pothole Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
