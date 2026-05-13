import tensorflow as tf
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model
import cv2
from ultralytics import YOLO
import os

# Initialize models as None
model = None
detection_model = None

# Path to your model
model_path = "MOH_full.keras"

# Load classification model
try:
    model = load_model(model_path)
    print("Classification model loaded successfully.")
except Exception as e:
    print(f"Error loading classification model: {e}")
    raise SystemExit("Failed to load classification model. Exiting application.")

# Load YOLO model for tumor detection
try:
    yolo_model_path = "best.pt"
    if os.path.exists(yolo_model_path):
        detection_model = YOLO(yolo_model_path)
        print("YOLO detection model loaded successfully.")
    else:
        print(f"YOLO model not found at {yolo_model_path}. Tumor detection disabled.")
        detection_model = None
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    detection_model = None

# Class names for classification
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def preprocess_image(img):
    """Preprocess image for classification model"""
    img = img.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return img_array

def draw_detections(image, detections):
    """Draw bounding boxes and labels on image with normal font"""
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    colors = {
        'glioma': (255, 0, 0),
        'meningioma': (0, 255, 0),
        'pituitary': (0, 0, 255),
        'tumor': (255, 255, 0)
    }
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        confidence = det['confidence']
        label = det['label']
        
        color = colors.get(label.lower(), (255, 0, 255))
        
        # Draw bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with NORMAL font (cv2.FONT_HERSHEY_PLAIN)
        label_text = f"{label}: {confidence:.1f}%"
        (label_width, label_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_PLAIN, 1.0, 1
        )
        cv2.rectangle(img_cv, (x1, y1 - label_height - 5), (x1 + label_width, y1), color, -1)
        cv2.putText(img_cv, label_text, (x1, y1 - 3), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
    
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return img_pil

def predict_image(img):
    """Main prediction function with YOLO detection"""
    if img is None or model is None:
        return img, {"Error": "Model failed to load"}, {}, "Error"
    
    try:
        yolo_detections = []
        yolo_has_tumor = False
        img_np = np.array(img)
        
        if detection_model:
            results = detection_model(img_np, conf=0.25, verbose=False)
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = float(box.conf[0]) * 100
                        class_id = int(box.cls[0])
                        label = detection_model.names[class_id] if hasattr(detection_model, 'names') else 'tumor'
                        
                        yolo_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'label': label
                        })
                        yolo_has_tumor = True
        
        if yolo_detections:
            annotated_img = draw_detections(img, yolo_detections)
        else:
            annotated_img = img
        
        processed_img = preprocess_image(img)
        predictions = model.predict(processed_img, verbose=0)
        
        classification_probs = {}
        if yolo_has_tumor:
            non_tumor_probs = [(i, p) for i, p in enumerate(predictions[0]) if class_names[i] != 'notumor']
            if non_tumor_probs:
                predicted_idx = max(non_tumor_probs, key=lambda x: x[1])[0]
            else:
                predicted_idx = np.argmax(predictions[0])
            
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
            result_text = f"**Tumor Detected!**\n\n**Classification:** {predicted_class.upper()}\n**Confidence:** {confidence:.2f}%"
        else:
            predicted_idx = np.argmax(predictions[0])
            predicted_class = class_names[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100
            result_text = f"**Classification:** {predicted_class.upper()}\n**Confidence:** {confidence:.2f}%"
        
        for cls, prob in zip(class_names, predictions[0]):
            classification_probs[cls] = float(prob * 100)
        
        return annotated_img, result_text, classification_probs, predicted_class.upper()
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(error_msg)
        return img, f"**Error:** {str(e)}", {}, "Error"

# Create Gradio interface with CLASSIC font styling
with gr.Blocks(
    title="Brain Tumor Detection System",
    theme=gr.themes.Soft(),
    css="""
        * {
            font-family: Arial, Helvetica, sans-serif !important;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: bold !important;
            text-transform: none !important;
            letter-spacing: normal !important;
        }
        button, .upload-button {
            font-family: Arial, Helvetica, sans-serif !important;
            font-weight: normal !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            border-radius: 5px !important;
        }
        .gr-textbox, .gr-markdown {
            font-family: Arial, Helvetica, sans-serif !important;
        }
    """
) as iface:
    gr.Markdown("""
    # Brain Tumor Detection System
    
    ### Dual-Model Analysis: YOLO + Classification
    
    Upload an MRI scan to detect and classify brain tumors using two powerful AI models:
    - **YOLO**: Detects and localizes tumors with bounding boxes
    - **EfficientNetV2**: Classifies tumor type (Glioma, Meningioma, Pituitary, or No Tumor)
    
    ### How it works:
    1. YOLO scans the image for tumor presence and draws bounding boxes
    2. CNN classifies the tumor type with confidence scores
    3. Results are combined for maximum accuracy
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload MRI")
            upload_btn = gr.UploadButton("Upload MRI Image", file_types=["image"])
            
            
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Detected")
            
            gr.Markdown("### Prediction")
            result_text = gr.Markdown(label="Diagnosis")
            
            gr.Markdown("### Class Probabilities")
            probabilities = gr.JSON(label="Confidence Scores")
            
            predicted_class_display = gr.Textbox(label="Final Diagnosis", interactive=False)
    
    input_image.change(
        fn=predict_image,
        inputs=[input_image],
        outputs=[output_image, result_text, probabilities, predicted_class_display]
    )
    
    upload_btn.upload(
        fn=lambda file: Image.open(file.name),
        inputs=[upload_btn],
        outputs=[input_image]
    ).then(
        fn=predict_image,
        inputs=[input_image],
        outputs=[output_image, result_text, probabilities, predicted_class_display]
    )
    
    gr.Markdown("""
    ---
    ### Important Notes:
    - **Glioma**: Most common type of brain tumor
    - **Meningioma**: Usually benign tumor
    - **Pituitary**: Tumor in the pituitary gland
    - **No Tumor**: Healthy brain scan
    
    *This system is for research purposes only. Always consult medical professionals for diagnosis.*
    """)

if __name__ == "__main__":
    iface.launch(share=True)